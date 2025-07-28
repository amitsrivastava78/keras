import keras
import keras.ops as ops
import numpy as np
from keras.layers import Dense, EinsumDense

from .quant import quantize


class GPTQ:
    def __init__(self, layer):
        self.original_layer = layer
        self.dev = "cpu"
        self.kernel_shape = layer.kernel.shape
        self.nsamples = 0

        # --- NEW, MORE ROBUST INITIALIZATION LOGIC ---
        if isinstance(layer, Dense):
            self.rows, self.columns = self.kernel_shape
            self.layer = layer

        elif isinstance(layer, EinsumDense):
            # For OPT-like models, attention layers use EinsumDense with a 3D kernel.
            if layer.kernel.ndim != 3:
                raise TypeError(
                    f"GPTQ only supports EinsumDense with 2D or 3D kernels, but got {layer.kernel.ndim}D."
                )

            shape = list(self.kernel_shape)
            # Heuristic: The largest dimension is the model's hidden size (d_model).
            # The other two are num_heads and head_dim.
            try:
                d_model_dim_index = shape.index(max(shape))
            except ValueError:
                raise TypeError(f"Could not determine hidden dimension from shape {shape}")

            if d_model_dim_index == 0:
                # Case: QKV projection. Kernel shape is (d_model, num_heads, head_dim).
                # The effective 2D weight matrix is (d_model, num_heads * head_dim).
                in_features, heads, head_dim = shape
                self.rows = in_features
                self.columns = heads * head_dim
            elif d_model_dim_index == 2:
                # Case: Attention Output projection. Kernel shape is (num_heads, head_dim, d_model).
                # The effective 2D weight matrix is (num_heads * head_dim, d_model).
                heads, head_dim, out_features = shape
                self.rows = heads * head_dim
                self.columns = out_features
            else:
                # This case (e.g., shape `(heads, d_model, head_dim)`) is not expected.
                raise TypeError(f"Unsupported 3D kernel arrangement in EinsumDense: {shape}")

            # Create a temporary object with a reshaped 2D kernel for the algorithm.
            self.layer = type('temp', (object,), {
                'kernel': ops.reshape(layer.kernel, (self.rows, self.columns)),
                'bias': layer.bias
            })()
        else:
            raise TypeError(f"Unsupported layer type for GPTQ: {type(layer)}")
        
        print("Hesssian Matrix shape is: ", self.rows, self.rows)

        # Initialize the Hessian with the *correct* number of input features (rows).
        self.H = ops.zeros((self.rows, self.rows), dtype="float32")

    def add_batch(self, inp, out=None):
        # This function should only accumulate data into the existing self.H matrix.
        if len(inp.shape) == 2:
            inp = ops.reshape(inp, (-1, inp.shape[-1]))
        inp = ops.cast(inp, "float32")

        # The Hessian `H` should already be initialized in `__init__`.
        # Let's ensure the input shape matches the expected dimension of H.
        if self.H.shape[0] != inp.shape[-1]:
            raise ValueError(
                f"Hessian matrix dimensions ({self.H.shape}) do not match "
                f"input dimensions ({inp.shape[-1]})."
            )

        # Perform the running average update for the Hessian
        if self.nsamples == 0:
            # For the first batch, just calculate the initial H
            self.H = ops.matmul(ops.transpose(inp), inp)
        else:
            # For subsequent batches, perform a weighted update
            self.H = self.H * (self.nsamples / (self.nsamples + inp.shape[0]))
            self.H += ops.matmul(ops.transpose(inp), inp)

        self.nsamples += inp.shape[0]

    def fasterquant(
        self, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False
    ):
        # W has shape (in_features, out_features), e.g., (768, 3072) for fc1
        W = ops.cast(self.layer.kernel, 'float32')
        # H has shape (in_features, in_features), e.g., (768, 768)
        H = ops.cast(self.H, 'float32')

        # Damping
        diag = ops.diagonal(H)
        damp = percdamp * ops.mean(diag)
        diag = diag + damp
        dead_mask = ops.equal(ops.diagonal(H), 0.0)
        diag = ops.where(dead_mask, 1.0, diag)
        H = H - ops.diag(ops.diagonal(H)) + ops.diag(diag)

        try:
            Hinv = ops.linalg.inv(H)
        except Exception:
            Hinv = ops.linalg.pinv(H)

        Q = ops.zeros_like(W)

        # --- Corrected and Stabilized Quantization Loop ---
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2]
            Q1 = ops.zeros_like(W1)
            Err1 = ops.zeros_like(W1)

            # Inner loop for in-block updates
            for i in range(count):
                w = W1[:, i]

                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                else:
                    self.quantizer.find_params(ops.expand_dims(w, 1), weight=True)

                q = quantize(
                    ops.expand_dims(w, 1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq
                )[:, 0]

                Q1 = ops.concatenate([Q1[:, :i], ops.expand_dims(q, 1), Q1[:, i+1:]], axis=1)
                err = w - q
                Err1 = ops.concatenate([Err1[:, :i], ops.expand_dims(err, 1), Err1[:, i+1:]], axis=1)

                # Propagate error to remaining columns *within the block*
                if i < count - 1:
                    W1_rem = W1[:, i+1:]
                    
                    # STABILIZED UPDATE RULE
                    v1 = ops.matmul(Hinv, ops.expand_dims(err, 1))
                    v2 = ops.matmul(ops.expand_dims(err, 0), W1_rem)
                    
                    # Normalization term to prevent exploding values
                    norm = ops.matmul(ops.expand_dims(err, 0), v1) + 1e-9 # Add epsilon
                    
                    update = ops.matmul(v1, v2) / norm
                    W1 = ops.concatenate([W1[:, :i+1], W1_rem - update], axis=1)

            # Update the full quantized matrix Q with the processed block
            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)
            
            # Propagate the block's error to the rest of the *entire* weight matrix
            if i2 < self.columns:
                W_rem_total = W[:, i2:]
                
                # STABILIZED UPDATE RULE for the rest of the matrix
                v1_total = ops.matmul(Hinv, Err1)
                v2_total = ops.matmul(ops.transpose(Err1), W_rem_total)
                
                # Normalization term
                norm_total = ops.trace(ops.matmul(ops.transpose(Err1), v1_total)) + 1e-9 # Add epsilon
                
                update_total = ops.matmul(v1_total, v2_total) / norm_total
                W = ops.concatenate([W[:, :i2], W_rem_total - update_total], axis=1)

        # Finalize the quantized weights
        if isinstance(self.original_layer, EinsumDense):
            Q = ops.reshape(Q, self.kernel_shape)

        new_weights = [ops.convert_to_numpy(Q)]
        if self.original_layer.bias is not None:
            new_weights.append(ops.convert_to_numpy(self.original_layer.bias))

        self.original_layer.set_weights(new_weights)

    def free(self):
        self.H = None
        keras.backend.clear_session()

