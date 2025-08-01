import random
import time

from datasets import load_dataset
import numpy as np
from tqdm import tqdm

import keras
from keras.src import ops
from keras.src import losses
from keras.src.layers import Dense, EinsumDense, Embedding
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

        # --- START OF FIX ---
        # Create the correct input structure based on the model type.
        inputs = None
        # Case 1: Standard KerasNLP model with a preprocessor.
        if hasattr(model, "preprocessor") and model.preprocessor is not None:
            inputs = {
                "token_ids": input_ids,
                "padding_mask": ops.ones_like(input_ids, dtype="bool"),
            }
        # Case 2: Custom or simple model without a preprocessor.
        else:
            # Use the model's actual input name as the key.
            # This makes the function compatible with the test model.
            inputs = input_ids
        # --- END OF FIX ---
        
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
    tokenizer, seqlen, dataset, nsamples=128, seed=0
):
    """
    Prepares and chunks the calibration dataloader, repeating short datasets.
    """
    all_tokens = []

    # --- Step 1: Unify all input types into a single list of tokens ---
    if isinstance(dataset, str):
        print(f"Loading '{dataset}' dataset from Hub...")
        if dataset == "wikitext2":
            d_name, d_config = "wikitext", "wikitext-2-raw-v1"
        elif dataset == "ptb":
            d_name, d_config = "ptb_text_only", "penn_treebank"
        # --- START OF C4-SPECIFIC FIX ---
        elif dataset == "c4":
            print("   -> Using memory-efficient streaming strategy for C4.")
            streaming_dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
            dataset_head = streaming_dataset.take(nsamples * 5)
            
            samples = []
            docs_for_sampling = list(dataset_head)

            for _ in range(nsamples):
                while True:
                    doc = random.choice(docs_for_sampling)
                    try:
                        # Call the tokenizer layer directly (the KerasNLP way)
                        # and squeeze the output to a 1D array.
                        tokenized_doc = np.squeeze(tokenizer(doc['text']))
                        if len(tokenized_doc) >= seqlen:
                            break
                    except Exception:
                        docs_for_sampling.remove(doc)
                        if not docs_for_sampling:
                            raise ValueError("Could not find enough valid documents in the C4 sample.")
                        continue
            
                j = random.randint(0, len(tokenized_doc) - seqlen - 1)
                sample_slice = tokenized_doc[j : j + seqlen]
                samples.append(np.reshape(sample_slice, (1, seqlen)))
            
            return np.array(samples, dtype=np.int32)
        # --- END OF C4-SPECIFIC FIX ---
        else:
            print(f"Warning: No specific alias found for '{dataset}'.")
            print(f"Attempting to load '{dataset}' directly with its default configuration.")
            d_name = dataset
            d_config = None # Use the default configuration for the dataset

        if d_name == "ptb_text_only":
            text_column = "sentence"
        else:
        # Default to "text" for wikitext2 and other datasets
            text_column = "text"
        
        raw_dataset = load_dataset(d_name, d_config, split="train")
        text_list = [d[text_column] for d in raw_dataset]
        full_text = "\n\n".join(text_list)
        all_tokens = tokenizer.tokenize(full_text)

    else:
        print("\n==> Using pre-made dataset/generator...")
        dataset_list = list(dataset)
        
        if not dataset_list:
            raise ValueError("Provided dataset is empty.")

        if isinstance(dataset_list[0], str):
            print("   (Dataset contains strings, tokenizing now...)")
            full_text = "\n\n".join(dataset_list)
            all_tokens = tokenizer.tokenize(full_text)
        else:
            print("   (Dataset is pre-tokenized, concatenating...)")
            concatenated_tokens = ops.concatenate(
                [ops.reshape(s, [-1]) for s in dataset_list], axis=0
            )
            all_tokens = ops.convert_to_numpy(concatenated_tokens)

    all_tokens = np.array(all_tokens, dtype=np.int32)

    # --- Step 2: Repeat data if it's too short ---
    required_tokens = nsamples * seqlen
    if len(all_tokens) < required_tokens:
        print(f"Warning: Dataset is too short ({len(all_tokens)} tokens). Repeating data to generate {nsamples} samples.")
        repeats = -(-required_tokens // len(all_tokens))  # Ceiling division
        all_tokens = np.tile(all_tokens, repeats)
    
    # --- Step 3: Chunk the token list into samples ---
    utils.set_random_seed(seed)
    
    calibration_samples = []
    for i in range(nsamples):
        start_index = i * seqlen
        end_index = start_index + seqlen
        sample = all_tokens[start_index:end_index]
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
    # --- START OF FIX ---
    # Initialize variables to None before the conditional logic to avoid the UnboundLocalError.
    embedding_layer = None
    transformer_blocks = []
    # --- END OF FIX ---
    if hasattr(model, "backbone"):
        print("   -> Detected KerasNLP model structure.")
        backbone = model.backbone
        transformer_blocks = backbone.transformer_layers
        # Find the embedding layer by checking for common names or by type.
        if hasattr(backbone, "token_embedding"):
            embedding_layer = backbone.token_embedding
        elif hasattr(backbone, "embedding"):
            embedding_layer = backbone.embedding
        else:
            raise ValueError("Could not automatically find an embedding layer in the model.")

    else:
        print("   -> Detected custom model structure.")
        for layer in model.layers:
            # The first Embedding layer found is assumed to be the main one.
            if isinstance(layer, Embedding) and embedding_layer is None:
                embedding_layer = layer
            # A "block" is a container-like layer with its own sub-layers
            # that we can quantize. This is a heuristic that works for the test.
            elif hasattr(layer, '_layers') and layer._layers:
                transformer_blocks.append(layer)

    if embedding_layer is None:
        raise ValueError("Could not automatically find an embedding layer in the model.")    
    if not transformer_blocks:
        raise ValueError("Could not automatically find any transformer-like blocks to quantize.")

    # Initial inputs are the outputs of the token embedding layer
    inputs = [
        embedding_layer(ops.convert_to_tensor(batch, dtype="int32"))
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

    # --- THE FIX ---
    # 1. Load ALL data needed from the generator/source in a single call.
    #    Request enough samples for both calibration and the test set.
    total_samples_to_request = config.nsamples + 50
    full_dataloader = get_dataloader(
        config.tokenizer, config.seqlen, config.dataset, nsamples=total_samples_to_request
    )

    # 2. Split the materialized data. This works because full_dataloader is now
    #    a NumPy array, which can be sliced and reused.
    calibration_dataloader = full_dataloader[:config.nsamples]
    test_dataloader = full_dataloader[config.nsamples:]
    
    tick = time.time()
    apply_gptq_layerwise(
        model,
        calibration_dataloader,  # Use the calibration slice
        len(calibration_dataloader), # Use the actual number of samples
        config.percdamp,
        config.groupsize,
        config.symmetric,
        config.act_order,
        config.wbits,
    )
    print(f"Total quantization time: {time.time() - tick:.2f} seconds")

    # Only run evaluation if there is data in the test set.
    # if test_dataloader.size > 0:
    #     print("\nLoading test data for evaluation...")
    #     calculate_perplexity(model, test_dataloader, config.seqlen)
    # else:
    #     print("\nSkipping perplexity evaluation: Not enough data to create a test set.")
    return model