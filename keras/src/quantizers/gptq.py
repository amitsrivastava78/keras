import keras
import keras.ops as ops
from keras.layers import Dense, EinsumDense
from .quant import quantize

class GPTQ:
    def __init__(self, layer):
        self.original_layer = layer
        self.layer = layer
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
            try:
                d_model_dim_index = shape.index(max(shape))
            except ValueError:
                raise TypeError(f"Could not determine hidden dimension from shape {shape}")

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
        W = ops.transpose(ops.cast(self.layer.kernel, 'float32'))
        H = ops.cast(self.H, 'float32')

        if actorder:
            perm = ops.argsort(-ops.diagonal(H))
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)

        diag_H = ops.diagonal(H)
        dead = ops.equal(diag_H, 0.0)
        diag_H = ops.where(dead, 1.0, diag_H)
        H = H + ops.diag(ops.where(dead, 1.0, ops.zeros_like(diag_H)))
        damp = percdamp * ops.mean(diag_H)
        diag_H = diag_H + damp
        H = (H - ops.diag(ops.diagonal(H))) + ops.diag(diag_H)

        Hinv = ops.linalg.inv(H)
        Q = ops.zeros_like(W)

        for i1 in range(0, self.rows, blocksize):
            i2 = min(i1 + blocksize, self.rows)
            count = i2 - i1

            W1 = W[:, i1:i2]
            Q1 = ops.zeros_like(W1)
            Err1 = ops.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
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

                if i < count - 1:
                    update = ops.matmul(ops.expand_dims(err, 1), ops.expand_dims(Hinv1[i, i+1:], 0))
                    W1 = ops.concatenate([W1[:, :i+1], W1[:, i+1:] - update], axis=1)

            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)

            if i2 < self.rows:
                update_total = ops.matmul(Err1, Hinv[i1:i2, i2:])
                W = ops.concatenate([W[:, :i2], W[:, i2:] - update_total], axis=1)
        
        if actorder:
            Q = ops.take(Q, invperm, axis=1)
        
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