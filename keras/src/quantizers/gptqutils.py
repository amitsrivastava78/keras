import random
import time

from datasets import load_dataset
from tqdm import tqdm

import keras
from keras.src import ops
from keras.src import losses
from keras.src.layers import Dense, EinsumDense
from keras.src.backend.common.global_state import clear_session as clear_session
from keras.src import utils

from .gptq import GPTQ
from .quant import Quantizer


def calculate_perplexity(model, dataloader, seqlen):
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

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            logits = outputs
        
        loss_fn = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = loss_fn(ops.expand_dims(targets, -1), logits)

        mask = ops.cast(ops.not_equal(targets, 1), dtype="float32")
        masked_loss = loss * mask

        total_nll += ops.sum(masked_loss)
        total_tokens += ops.sum(mask)

    if total_tokens == 0:
        print("Warning: No tokens were evaluated.")
        return float("inf")

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

    utils.set_random_seed(seed)

    dataset = load_dataset(d_name, d_config, split="train")
    text_list = [d["text"] for d in dataset]
    full_text = "\n\n".join(text_list)
    tokenized_text = tokenizer.tokenize(full_text)
    tokenized_text = ops.convert_to_numpy(tokenized_text)

    calibration_samples = []
    for _ in range(nsamples):
        i = random.randint(0, len(tokenized_text) - seqlen - 1)
        sample = tokenized_text[i : i + seqlen]
        calibration_samples.append(ops.reshape(sample, (1, seqlen)))
    final_array = ops.stack(calibration_samples, axis=0)
    return ops.convert_to_numpy(final_array)


def _find_layers_recursive(layer, prefix, found_layers):
    """
    Recursively search for Dense and EinsumDense layers and record them.
    """
    for sub_layer in layer._layers:
        # Construct a unique name for the layer based on its hierarchy
        layer_name = f"{prefix}.{sub_layer.name}"
        if isinstance(sub_layer, (Dense, EinsumDense)):
            found_layers[layer_name] = sub_layer
        
        # Recurse into nested layers that are not the target types
        elif hasattr(sub_layer, '_layers') and sub_layer._layers:
            _find_layers_recursive(sub_layer, layer_name, found_layers)


def find_layers_in_block(block):
    """
    A pluggable, generic function to find all Dense and EinsumDense layers
    within any transformer block by using a recursive search.
    """
    found_layers = {}
    # Start the recursive search from the block itself
    _find_layers_recursive(block, "block", found_layers)
    return found_layers


def apply_gptq_layerwise(
    model,
    dataloader,
    nsamples,
    seqlen,
    percdamp,
    groupsize,
    symmetric,
    act_order,
    wbits,
):
    """
    Performs sequential, model-agnostic quantization by dynamically finding
    layers and capturing their inputs via hooks.
    """
    # The 'seqlen' parameter is currently unused but retained for future extensions.
    print("Starting model quantization...")
    backbone = model.backbone
    transformer_blocks = backbone.transformer_layers

    # Initial inputs are the outputs of the token embedding layer
    inputs = [
        backbone.token_embedding(ops.convert_to_tensor(batch[0], dtype="int32"))
        for batch in dataloader
    ]

    for i, block in enumerate(transformer_blocks):
        print(f"\n--- Quantizing Block {i} ---")

        sub_layers_map = find_layers_in_block(block)
        if not sub_layers_map:
            print(f"  No Dense or EinsumDense layers found in block {i}. Skipping.")
        else:
            print(f"  Found layers: {list(sub_layers_map.keys())}")
            gptq_objects = {name: GPTQ(layer) for name, layer in sub_layers_map.items()}

            # --- START OF FIX ---
            # Initialize dictionaries outside the try block to ensure they are in scope for the `finally` block.
            captured_inputs = {name: [] for name in sub_layers_map.keys()}
            original_calls = {}
            # --- END OF FIX ---

            def create_hook(name, original_call_func):
                """A factory for creating a hook to capture layer inputs."""
                def hook(*args, **kwargs):
                    if args:
                        inp = args[0]
                    else:
                        inp = kwargs["inputs"]
                    captured_inputs[name].append(inp)
                    return original_call_func(*args, **kwargs)
                return hook

            try:
                for name, layer in sub_layers_map.items():
                    original_call = layer.call
                    original_calls[name] = original_call
                    layer.call = create_hook(name, original_call)

                print(f"Capturing activations for block {i}...")
                for j in range(nsamples):
                    current_input = inputs[j]
                    if len(current_input.shape) == 2:
                        current_input = ops.expand_dims(current_input, axis=0)
                    _ = block(current_input)

            finally:
                for name, layer in sub_layers_map.items():
                    if name in original_calls:
                        layer.call = original_calls[name]

            print(f"Building Hessians for block {i}...")
            for name, gptq_object in gptq_objects.items():
                layer_inputs = ops.concatenate(captured_inputs[name], axis=0)

                # --- START OF FIX ---
                # Explicitly reshape the input tensor to be 2D, with the second
                # dimension matching the number of input features expected by the layer's kernel.
                # This correctly handles inputs of any dimensionality (e.g., 3D or 4D).
                num_features = gptq_object.rows
                inp_reshaped = ops.reshape(layer_inputs, (-1, num_features))
                # --- END OF FIX ---
                gptq_object.add_batch(inp_reshaped)

            quantizer = Quantizer()
            quantizer.configure(
                wbits, perchannel=True, sym=symmetric, groupsize=groupsize
            )
            for name, gptq_object in gptq_objects.items():
                print(f"  Quantizing {name}...")
                gptq_object.quantizer = quantizer
                gptq_object.fasterquant(
                    percdamp=percdamp, groupsize=groupsize, actorder=act_order
                )
                gptq_object.free()

            del gptq_objects, captured_inputs, original_calls

        if i < len(transformer_blocks) - 1:
            print(f"Generating inputs for block {i + 1}...")
            next_block_inputs = []
            for j in range(nsamples):
                current_input = inputs[j]
                if len(current_input.shape) == 2:
                    current_input = ops.expand_dims(current_input, axis=0)
                output = block(current_input)[0]
                next_block_inputs.append(output)
            inputs = next_block_inputs

        keras.backend.clear_session()

    print("\nQuantization process complete.")

def quantize_model(model, config):
    """
    Top-level function to quantize a Keras model using GPTQ.
    """
    print("Starting GPTQ quantization process...")

    dataloader = get_dataloader(
        config.tokenizer, config.seqlen, config.dataset, config.nsamples
    )

    tick = time.time()
    apply_gptq_layerwise(
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

    print("\nLoading test data for evaluation...")
    test_dataloader = get_dataloader(
        config.tokenizer, config.seqlen, config.dataset, nsamples=50
    )
    calculate_perplexity(model, test_dataloader, config.seqlen)

    return model