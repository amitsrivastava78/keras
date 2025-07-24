import random
import numpy as np
import argparse
import keras
import keras.ops as ops
from transformers import AutoTokenizer
from datasets import load_dataset
from gptq import GPTQ
from quant import Quantizer
import time
from tqdm import tqdm

def eval_keras(model, dataloader, seqlen):
    """Evaluation loop for Perplexity."""
    print('\nEvaluating perplexity...')
    total_nll = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Evaluating PPL"):
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]

        outputs = model(input_ids)
        logits = outputs.logits

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=None)
        loss = loss_fn(ops.expand_dims(targets, -1), logits)

        mask = ops.cast(ops.not_equal(targets, 1), dtype='float32')
        masked_loss = loss * mask

        total_nll += ops.sum(masked_loss)
        total_tokens += ops.sum(mask)

    if total_tokens == 0:
        print("Warning: No tokens were evaluated.")
        return float('inf')

    ppl = ops.exp(total_nll / total_tokens)
    print(f"\nFinal Perplexity: {ppl:.4f}")
    return ppl

def get_dataloader(tokenizer, seqlen, dataset_name="wikitext2", nsamples=128, seed=0):
    """Prepares the calibration dataloader with RANDOM SAMPLING."""
    print(f"Loading '{dataset_name}' dataset for calibration...")
    if dataset_name == 'wikitext2':
        d_name, d_config = 'wikitext', 'wikitext-2-raw-v1'
    elif dataset_name == 'ptb':
        d_name, d_config = 'ptb_text_only', 'penn_treebank'
    else:
        d_name, d_config = 'c4', 'en'

    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed) # Retained for reproducibility with tf.data
    keras.utils.set_random_seed(seed)

    dataset = load_dataset(d_name, d_config, split="train")
    text_list = [d['text'] for d in dataset]
    full_text = "\n\n".join(text_list)
    tokenized_text = tokenizer(full_text, return_tensors='np').input_ids[0]

    calibration_samples = []
    for _ in range(nsamples):
        i = random.randint(0, len(tokenized_text) - seqlen - 1)
        sample = tokenized_text[i:i + seqlen]
        calibration_samples.append(np.reshape(sample, (1, seqlen)))

    # dataloader = tf.data.Dataset.from_generator(
    #     lambda: calibration_samples,
    #     output_signature=tf.TensorSpec(shape=(1, seqlen), dtype=tf.int64)
    # )
    dataloader = np.array(calibration_samples, dtype=np.int32)
    return dataloader

def sequential_keras(model, dataloader, nsamples, seqlen, percdamp, groupsize, symmetric, act_order, wbits):
    """
    Performs quantization of the Keras model.

    """
    print("Starting model quantization...")

    # Retrieve the list of decoder layers from the model
    decoder = model.model.decoder
    layers = decoder.layers
    
    # Get the initial embeddings from the model
    print("Getting initial embeddings...")
    # The inputs to the first block are the embeddings for each calibration sample
    inputs = [decoder.embed_tokens(batch) for batch in dataloader]
    
    # This list will hold the outputs of the current layer, which become the
    # inputs for the next layer's Hessian calculation.
    next_layers_inputs = inputs

    # Sequentially quantize each decoder layer
    for i in range(len(layers)):
        print(f"\n--- Quantizing Block {i} ---")
        layer = layers[i]

        # Define the sub-layers within the current block
        sub_layers = {
            'self_attn.q_proj': layer.self_attn.q_proj,
            'self_attn.k_proj': layer.self_attn.k_proj,
            'self_attn.v_proj': layer.self_attn.v_proj,
            'self_attn.out_proj': layer.self_attn.out_proj,
            'fc1': layer.fc1,
            'fc2': layer.fc2
        }

        # Create a GPTQ object for each sub-layer
        gptq_objects = {}
        for name, sub_layer in sub_layers.items():
            gptq_objects[name] = GPTQ(sub_layer)
            quantizer = Quantizer()
            quantizer.configure(wbits, perchannel=True, sym=symmetric)
            gptq_objects[name].quantizer = quantizer

        # --- Feed Calibration Data to Build Hessians ---
        print(f"Building Hessians for block {i}...")
                # --- Feed Calibration Data to Build Hessians ---
        print(f"Building Hessians for block {i}...")

        def get_intermediate_inputs(block_input, current_layer):
            """Simulates the forward pass to get correct inputs for each dense layer."""
            # Ensure block_input is 3D for the layers
            attn_input = current_layer.self_attn_layer_norm(block_input)
            attn_output = current_layer.self_attn(attn_input)[0]
            fc1_input = current_layer.final_layer_norm(block_input + attn_output)
            fc2_input = current_layer.activation_fn(current_layer.fc1(fc1_input))
            return {
                'self_attn.q_proj': attn_input, 'self_attn.k_proj': attn_input,
                'self_attn.v_proj': attn_input, 'self_attn.out_proj': attn_output,
                'fc1': fc1_input, 'fc2': fc2_input
            }
        for j in range(nsamples):
            # Pass the 3D tensor (e.g., shape (1, 512, 768)) directly to the helper
            current_input = inputs[j]
            intermediate_inputs = get_intermediate_inputs(current_input, layer)
            for name, gptq_object in gptq_objects.items():
                # Reshape the 3D activations to 2D for the Hessian calculation
                inp_3d = intermediate_inputs[name]
                # inp_2d = tf.reshape(inp_3d, [-1, tf.shape(inp_3d)[-1]])
                inp_2d = keras.ops.reshape(inp_3d, (-1, keras.ops.shape(inp_3d)[-1]))
                gptq_object.add_batch(inp_2d, None)

        # --- Quantize Each Sub-layer in the Block ---
        for name, gptq_object in gptq_objects.items():
            print(f"  Quantizing {name}...")
            gptq_object.fasterquant(
                blocksize=128, percdamp=percdamp, groupsize=groupsize, actorder=act_order
            )
            gptq_object.free()

        # --- Generate Inputs for the NEXT Block ---
        if i < len(layers) - 1:
            print(f"Generating inputs for block {i + 1}...")
            next_block_inputs = []
            for j in range(nsamples):
                output = layer(inputs[j])[0]
                next_block_inputs.append(output)
            inputs = next_block_inputs

    print('\nQuantization process complete.')
    return {}


def quantize_model(model, dataset, nsamples, seqlen, percdamp, groupsize, symmetric, act_order, wbits):
    

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    dataloader = get_dataloader(tokenizer, seqlen, dataset, nsamples)

    tick = time.time()
    sequential_keras(model, dataloader, nsamples, seqlen, percdamp, groupsize, symmetric, act_order, wbits)
    print(f"Total quantization time: {time.time() - tick:.2f} seconds")

    print("\nLoading test data for evaluation...")
    test_dataloader = get_dataloader(tokenizer, seqlen, dataset, nsamples=50) 
    eval_keras(model, test_dataloader, seqlen)