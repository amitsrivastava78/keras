import random
import time

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import keras
import keras.ops as ops

# Assuming these are local modules adapted for Keras 3.0
from .gptq import GPTQ
from .quant import Quantizer


def eval_keras(model, dataloader, seqlen):
    """
    Evaluation loop for Perplexity using Keras 3.0.

    This function calculates the perplexity of a model on a given dataset.
    It is backend-agnostic, relying on `keras.ops` for computations.
    """
    print("\nEvaluating perplexity...")
    total_nll = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating PPL"):
        batch = ops.convert_to_tensor(batch, dtype="int32")

        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        inputs = {
            "token_ids": input_ids,
            "padding_mask": ops.ones_like(input_ids, dtype="bool"),
        }
        
        outputs = model(inputs)

        # --- THIS IS THE FIX ---
        # Check if the model output is a dictionary or a direct tensor
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
        
        # Calculate loss using Keras API
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = loss_fn(ops.expand_dims(targets, -1), logits)

        # Create a mask to ignore padding tokens (assuming padding token ID is 1)
        mask = ops.cast(ops.not_equal(targets, 1), dtype="float32")
        masked_loss = loss * mask

        # Accumulate total negative log-likelihood and token count
        total_nll += ops.sum(masked_loss)
        total_tokens += ops.sum(mask)

    if total_tokens == 0:
        print("Warning: No tokens were evaluated.")
        return float("inf")

    # Calculate perplexity
    ppl = ops.exp(total_nll / total_tokens)
    print(f"\nFinal Perplexity: {float(ppl):.4f}")
    return ppl


def get_dataloader(
    tokenizer, seqlen, dataset_name="wikitext2", nsamples=128, seed=0
):
    """
    Prepares the calibration dataloader with RANDOM SAMPLING.

    This function is now fully backend-agnostic and returns a NumPy array,
    which is compatible with all Keras backends.
    """
    print(f"Loading '{dataset_name}' dataset for calibration...")
    if dataset_name == "wikitext2":
        d_name, d_config = "wikitext", "wikitext-2-raw-v1"
    elif dataset_name == "ptb":
        d_name, d_config = "ptb_text_only", "penn_treebank"
    else:
        d_name, d_config = "c4", "en"

    # Set random seeds for reproducibility across libraries
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

    # Load dataset and tokenize
    dataset = load_dataset(d_name, d_config, split="train")
    text_list = [d["text"] for d in dataset]
    full_text = "\n\n".join(text_list)
    tokenized_text = tokenizer.tokenize(full_text)
    tokenized_text = ops.convert_to_numpy(tokenized_text)

    # Create calibration samples by random sampling
    calibration_samples = []
    for _ in range(nsamples):
        i = random.randint(0, len(tokenized_text) - seqlen - 1)
        sample = tokenized_text[i : i + seqlen]
        calibration_samples.append(np.reshape(sample, (1, seqlen)))

    # Return as a NumPy array for universal compatibility
    return np.array(calibration_samples, dtype=np.int32)


def sequential_keras(
    model, dataloader, nsamples, seqlen, percdamp, groupsize, symmetric, act_order, wbits
):
    """Performs sequential quantization of a Keras model."""
    print("Starting model quantization...")

    if not hasattr(model, "backbone"):
        raise ValueError("Model must have a `backbone` attribute.")
    backbone = model.backbone
    layers = backbone.transformer_layers

    print("Getting initial embeddings...")
    inputs = [
        backbone.token_embedding(ops.convert_to_tensor(batch[0], dtype="int32"))
        for batch in dataloader
    ]

    for i in range(len(layers)):
        print(f"\n--- Quantizing Block {i} ---")
        layer = layers[i]

        sub_layers_map = {
            "self_attn.q_proj": layer._self_attention_layer._query_dense,
            "self_attn.k_proj": layer._self_attention_layer._key_dense,
            "self_attn.v_proj": layer._self_attention_layer._value_dense,
            "self_attn.out_proj": layer._self_attention_layer._output_dense,
            "fc1": layer._feedforward_intermediate_dense,
            "fc2": layer._feedforward_output_dense,
        }
        
        gptq_objects = {}
        for name, sub_layer in sub_layers_map.items():
            if sub_layer is None: continue
            gptq_objects[name] = GPTQ(sub_layer)
            quantizer = Quantizer()
            quantizer.configure(wbits, perchannel=True, sym=symmetric)
            gptq_objects[name].quantizer = quantizer

        def get_intermediate_inputs_for_block(block_input, current_layer):
            if len(block_input.shape) == 2:
                block_input = ops.expand_dims(block_input, axis=0)

            attn_qkv_input = current_layer._self_attention_layer_norm(block_input)
            
            attention_layer = current_layer._self_attention_layer
            query = attention_layer._query_dense(attn_qkv_input)
            key = attention_layer._key_dense(attn_qkv_input)
            value = attention_layer._value_dense(attn_qkv_input)
            
            attn_out_input, _ = attention_layer._compute_attention(query, key, value)
            
            attn_output = attention_layer._output_dense(attn_out_input)
            residual = block_input + attn_output

            fc1_input = current_layer._feedforward_layer_norm(residual)
            
            fc2_input = current_layer.activation(
                current_layer._feedforward_intermediate_dense(fc1_input)
            )
            
            return attn_qkv_input, attn_qkv_input, attn_qkv_input, attn_out_input, fc1_input, fc2_input

        block_input_tensor = keras.Input(shape=inputs[0].shape[1:], batch_size=1, dtype=inputs[0].dtype)
        outputs_of_interest = get_intermediate_inputs_for_block(block_input_tensor, layer)
        temp_model = keras.Model(inputs=block_input_tensor, outputs=outputs_of_interest)
        
        sub_layer_names = list(sub_layers_map.keys())

        print(f"Building Hessians for block {i}...")
        for j in range(nsamples):
            current_input = inputs[j]
            intermediate_inputs = temp_model(current_input)
            
            for name_idx, name in enumerate(sub_layer_names):
                inp = intermediate_inputs[name_idx]
                
                # --- THIS IS THE FIX ---
                # Reshape the input tensor correctly for each layer type.
                if name == "self_attn.out_proj":
                    # The input to out_proj is 4D: (batch, seq_len, num_heads, head_dim)
                    # We need to reshape it to (batch * seq_len, num_heads * head_dim) to match the kernel.
                    in_shape = ops.shape(inp) # e.g., (1, 128, 12, 64)
                    inp_reshaped = ops.reshape(inp, (in_shape[0] * in_shape[1], in_shape[2] * in_shape[3]))
                else:
                    # All other layers have 3D input: (batch, seq_len, features)
                    # We reshape to (batch * seq_len, features)
                    inp_reshaped = ops.reshape(inp, (-1, ops.shape(inp)[-1]))

                gptq_objects[name].add_batch(inp_reshaped)


        for name, gptq_object in gptq_objects.items():
            print(f"  Quantizing {name}...")
            gptq_object.fasterquant(percdamp=percdamp, groupsize=groupsize, actorder=act_order)
            gptq_object.free()

            if i < len(layers) - 1:
                print(f"Generating inputs for block {i + 1}...")
                next_block_inputs = []
                for j in range(nsamples):
                    # --- DEFENSIVE SHAPE CORRECTION ---
                    # Force the input to be 3D before passing it to the quantized layer.
                    current_input = inputs[j]
                    if len(current_input.shape) == 2:
                        current_input = ops.expand_dims(current_input, axis=0)
                    
                    output = layer(current_input)[0]
                    next_block_inputs.append(output)
                inputs = next_block_inputs
        
        del gptq_objects, temp_model
        if 'next_block_inputs' in locals():
            del next_block_inputs
        keras.backend.clear_session()

    print("\nQuantization process complete.")


def quantize_model(model, config):
    """
    Top-level function to quantize a Keras model using GPTQ.
    """
    print("Starting GPTQ quantization process...")

    # Get calibration data
    dataloader = get_dataloader(
        config.tokenizer, config.seqlen, config.dataset, config.nsamples
    )

    # Perform sequential quantization
    tick = time.time()
    sequential_keras(
        model,
        dataloader,
        config.nsamples,
        config.seqlen,
        config.percdamp,
        config.groupsize,
        config.symmetric,
        config.act_order,
        config.wbits,
    )
    print(f"Total quantization time: {time.time() - tick:.2f} seconds")

    # Evaluate the quantized model
    print("\nLoading test data for evaluation...")
    test_dataloader = get_dataloader(
        config.tokenizer, config.seqlen, config.dataset, nsamples=50
    )
    eval_keras(model, test_dataloader, config.seqlen)

    return model
