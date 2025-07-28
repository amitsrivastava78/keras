import keras
import keras.ops as ops
import numpy as np
from keras.layers import Dense, EinsumDense

from .quant import quantize


# In gptq.py

class GPTQ:
    def __init__(self, layer):
        self.original_layer = layer
        self.layer = layer 
        
        # --- THIS IS THE FIX ---
        # Store the original kernel shape for the final reshape operation.
        self.kernel_shape = layer.kernel.shape
        
        W = ops.cast(layer.kernel, 'float32')
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        
        self.H = ops.zeros((self.rows, self.rows), dtype="float32")
        self.nsamples = 0
        self.quantizer = None
        
        if isinstance(layer, EinsumDense):
            if layer.kernel.ndim != 3:
                raise TypeError(f"Unsupported EinsumDense kernel ndim: {layer.kernel.ndim}")
            shape = list(layer.kernel.shape)
            d_model_dim_index = shape.index(max(shape))
            if d_model_dim_index == 0:
                in_features, heads, head_dim = shape
                self.rows, self.columns = in_features, heads * head_dim
            elif d_model_dim_index == 2:
                heads, head_dim, out_features = shape
                self.rows, self.columns = heads * head_dim, out_features
            else:
                raise TypeError(f"Unsupported EinsumDense shape: {shape}")
            self.H = ops.zeros((self.rows, self.rows), dtype="float32")
            self.layer = type('temp', (object,), {
                'kernel': ops.reshape(layer.kernel, (self.rows, self.columns)),
                'bias': layer.bias
            })()


    def add_batch(self, inp):
        if len(inp.shape) > 2:
            inp = ops.reshape(inp, (-1, inp.shape[-1]))
        inp = ops.cast(inp, "float32")

        if self.H.shape[0] != inp.shape[-1]:
            raise ValueError(
                f"Hessian dimensions ({self.H.shape[0]}) do not match input features ({inp.shape[-1]})."
            )
        
        # The paper's formula is H = 2 * E[X^T X]
        current_H = 2 * ops.matmul(ops.transpose(inp), inp)
        
        if self.nsamples == 0:
            self.H = current_H
        else:
            self.H = self.H * (self.nsamples / (self.nsamples + inp.shape[0]))
            self.H += current_H * (inp.shape[0] / (self.nsamples + inp.shape[0]))
        self.nsamples += inp.shape[0]

    def fasterquant(
        self, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False
    ):
        # --- This is a direct and correct port of the working reference algorithm ---
        W = ops.transpose(ops.cast(self.layer.kernel, 'float32')) # Shape: (out_features, in_features)
        H = ops.cast(self.H, 'float32') # Shape: (in_features, in_features)

        # Damping
        diag = ops.diagonal(H)
        damp = percdamp * ops.mean(diag)
        diag = diag + damp
        dead_mask = ops.equal(ops.diagonal(H), 0.0)
        diag = ops.where(dead_mask, 1.0, diag)
        H = H - ops.diag(ops.diagonal(H)) + ops.diag(diag)

        Hinv = ops.linalg.inv(H)
        Q = ops.zeros_like(W)

        # The loops iterate over INPUT features
        for i1 in range(0, self.rows, blocksize):
            i2 = min(i1 + blocksize, self.rows)
            count = i2 - i1

            W1 = W[:, i1:i2]
            Q1 = ops.zeros_like(W1)
            Err1 = ops.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2] # This slice is now dimensionally correct

            for i in range(count):
                w = W1[:, i] # w represents all connections FROM an input feature
                d = Hinv1[i, i]

                if groupsize != -1:
                    # Grouping is applied to input features
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                else:
                    self.quantizer.find_params(ops.expand_dims(w, 1), weight=True)

                q = quantize(
                    ops.expand_dims(w, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                )[:, 0]

                Q1 = ops.concatenate([Q1[:, :i], ops.expand_dims(q, 1), Q1[:, i+1:]], axis=1)
                err = (w - q) / d
                Err1 = ops.concatenate([Err1[:, :i], ops.expand_dims(err, 1), Err1[:, i+1:]], axis=1)

                # In-block update
                if i < count - 1:
                    update = ops.matmul(ops.expand_dims(err, 1), ops.expand_dims(Hinv1[i, i+1:], 0))
                    W1 = ops.concatenate([W1[:, :i+1], W1[:, i+1:] - update], axis=1)

            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)

            # Outer-block update
            if i2 < self.rows:
                update_total = ops.matmul(Err1, Hinv[i1:i2, i2:])
                W = ops.concatenate([W[:, :i2], W[:, i2:] - update_total], axis=1)
        
        # Finalize and set weights, transposing back to original (in, out) shape
        Q = ops.transpose(Q)
        
        if isinstance(self.original_layer, EinsumDense):
            Q = ops.reshape(Q, self.kernel_shape)

        new_weights = [ops.convert_to_numpy(Q)]
        if self.original_layer.bias is not None:
            new_weights.append(ops.convert_to_numpy(self.original_layer.bias))

        self.original_layer.set_weights(new_weights)

    def free(self):
        self.H = None
        keras.backend.clear_session()

