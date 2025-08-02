import os
import pickle
from collections import namedtuple
from unittest import mock
import keras
import keras_nlp
import sys
import keras

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src import testing
from keras.src import tree
from keras.src.layers.core.input_layer import Input

from keras.src.models.functional import Functional
from keras.src.models.model import Model
from keras.src.models.model import model_from_json
from keras.src.quantizers.gptqconfig import GPTQConfig
from transformers import AutoTokenizer, TFAutoModelForCausalLM


from tqdm import tqdm
from keras.src import ops


def calculate_perplexity(model, dataloader):
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

def _get_model():
    input_a = Input(shape=(3,), batch_size=2, name="input_a")
    input_b = Input(shape=(3,), batch_size=2, name="input_b")
    x = input_a + input_b
    x = layers.Dense(5)(x)
    outputs = layers.Dense(4)(x)
    model = Model([input_a, input_b], outputs)
    return model


def _get_model_multi_outputs_list():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    output_b = layers.Dense(1, name="output_b", activation="sigmoid")(x)
    model = Model(x, [output_a, output_b])
    return model


def _get_model_multi_outputs_list_no_output_names():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1)(x)
    output_b = layers.Dense(1, activation="sigmoid")(x)
    model = Model(x, [output_a, output_b])
    return model


def _get_model_single_output():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    model = Model(x, output_a)
    return model


def _get_model_single_output_list():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    model = Model(x, [output_a])
    return model


def _get_model_single_output_dict():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    model = Model(x, {"output_a": output_a})
    return model


def _get_model_multi_outputs_dict():
    x = Input(shape=(3,), name="input_a")
    output_a = layers.Dense(1, name="output_a")(x)
    output_b = layers.Dense(1, name="output_b", activation="sigmoid")(x)
    model = Model(x, {"output_a": output_a, "output_b": output_b})
    return model


def _get_model_multi_outputs_struct_list_like(_type):
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    model = Model(x, _type([y1, y2]))
    return model


def _get_model_multi_outputs_struct_namedtuple():
    Y = namedtuple("Y", ["y1", "y2"])
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    model = Model(x, Y(y1, y2))
    return model, Y


def _get_model_multi_outputs_struct_dict():
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    model = Model(x, {"a": y1, "b": y2})
    return model


def _get_model_multi_outputs_struct():
    x = Input(shape=(3,), name="x")
    y1 = layers.Dense(1, name="y1", activation="sigmoid")(x)
    y2 = layers.Dense(1, name="y2", activation="sigmoid")(x)
    y3 = layers.Dense(1, name="y3", activation="sigmoid")(x)
    model = Model(
        x,
        {
            "a": (y1, y2),
            "b": {"b1": y1, "b2": y2},
            "c": {"c1": (y1, y2), "c2": y2},
            "d": y3,
        },
    )
    return model


def _get_model_multi_outputs_dict_with_single_tensor():
    x = Input(shape=(3,), name="input_a")
    output = layers.Dense(1, name="output_a")(x)
    model = Model(x, {"output_a": output, "output_b": output})
    return model


def _get_model_with_custom_compute_loss():
    class MyModel(Model):
        def __init__(self):
            inputs = Input(shape=(3,), name="inputs")
            outputs = layers.Dense(1, name="a")(inputs)
            super().__init__(inputs=inputs, outputs=outputs)

        def compute_loss(self, x, y, y_pred, sample_weight=None, **kwargs):
            y_pred = [y_pred, y_pred]  # To list
            return super().compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, **kwargs
            )

    model = MyModel()
    return model


def _get_model_with_duplicate_variable_path():
    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.dense1 = layers.Dense(4, activation="relu", name="layer1")
            self.dense2 = layers.Dense(4, activation="relu", name="layer1")
            self.dense3 = layers.Dense(2)

        def call(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            return self.dense3(x)

    model = MyModel()
    x = np.random.random((1, 16))
    model(x)
    return model


def _get_variable_value_by_path(variables, path):
    for v in variables:
        if v.path == path:
            return v.value
    raise ValueError(f"No variable was find with path = {path}")


@pytest.mark.requires_trainable_backend
class ModelTest(testing.TestCase):
    def test_functional_rerouting(self):
        model = _get_model()
        self.assertIsInstance(model, Functional)

    def test_json_serialization(self):
        model = _get_model()
        json_string = model.to_json()
        new_model = model_from_json(json_string)
        self.assertEqual(json_string, new_model.to_json())

    def test_tuple_input_model_subclass(self):
        # https://github.com/keras-team/keras/issues/324

        class MultiInputModel(Model):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.dense1 = layers.Dense(4)

            def call(self, inputs):
                a, b = inputs
                r = self.dense1(a)
                return layers.concatenate([r, b])

        model = MultiInputModel()
        x1 = np.random.rand(3, 3)
        x2 = np.random.rand(3, 2)
        out = model((x1, x2))
        self.assertEqual(out.shape, (3, 6))

    def test_reviving_functional_from_config_custom_layer(self):
        class CustomDense(layers.Layer):
            def __init__(self, units, **kwargs):
                super().__init__(**kwargs)
                self.dense = layers.Dense(units)

            def call(self, x):
                return self.dense(x)

        inputs = layers.Input((4,))
        outputs = CustomDense(10)(inputs)
        model = Model(inputs, outputs)
        config = model.get_config()

        new_model = Model.from_config(
            config, custom_objects={"CustomDense": CustomDense}
        )
        self.assertIsInstance(new_model, Functional)

    def test_reviving_functional_from_config_custom_model(self):
        class CustomModel(Model):
            def __init__(self, *args, param=1, **kwargs):
                super().__init__(*args, **kwargs)
                self.param = param

            def get_config(self):
                base_config = super().get_config()
                config = {"param": self.param}
                return base_config | config

        inputs = layers.Input((3,))
        outputs = layers.Dense(5)(inputs)
        model = CustomModel(inputs=inputs, outputs=outputs, param=3)

        new_model = CustomModel.from_config(model.get_config())
        self.assertEqual(new_model.param, 3)

    @parameterized.named_parameters(
        ("single_output_1", _get_model_single_output),
        ("single_output_2", _get_model_single_output),
        ("single_output_3", _get_model_single_output),
        ("single_output_4", _get_model_single_output),
        ("single_list_output_1", _get_model_single_output_list),
        ("single_list_output_2", _get_model_single_output_list),
        ("single_list_output_3", _get_model_single_output_list),
        ("single_list_output_4", _get_model_single_output_list),
    )
    def test_functional_pickling(self, model_fn):
        model = model_fn()
        self.assertIsInstance(model, Functional)
        model.compile()
        x = np.random.rand(8, 3)

        reloaded_pickle = pickle.loads(pickle.dumps(model))

        pred_reloaded = reloaded_pickle.predict(x)
        pred = model.predict(x)

        self.assertAllClose(np.array(pred_reloaded), np.array(pred))

    @parameterized.named_parameters(
        ("single_output_1", _get_model_single_output, None),
        ("single_output_2", _get_model_single_output, "list"),
        ("single_output_3", _get_model_single_output, "dict"),
        ("single_output_4", _get_model_single_output, "dict_list"),
        ("single_list_output_1", _get_model_single_output_list, None),
        ("single_list_output_2", _get_model_single_output_list, "list"),
        ("single_list_output_3", _get_model_single_output_list, "dict"),
        ("single_list_output_4", _get_model_single_output_list, "dict_list"),
        ("single_dict_output_1", _get_model_single_output_dict, None),
        ("single_dict_output_2", _get_model_single_output_dict, "list"),
        ("single_dict_output_3", _get_model_single_output_dict, "dict"),
        ("single_dict_output_4", _get_model_single_output_dict, "dict_list"),
    )
    def test_functional_single_output(self, model_fn, loss_type):
        model = model_fn()
        self.assertIsInstance(model, Functional)
        loss = "mean_squared_error"
        if loss_type == "list":
            loss = [loss]
        elif loss_type == "dict":
            loss = {"output_a": loss}
        elif loss_type == "dict_list":
            loss = {"output_a": [loss]}
        model.compile(
            optimizer="sgd",
            loss=loss,
            metrics={
                "output_a": ["mean_squared_error", "mean_absolute_error"],
            },
            weighted_metrics={
                "output_a": "mean_squared_error",
            },
        )
        # Fit the model to make sure compile_metrics are built
        x = np.random.rand(8, 3)
        y = np.random.rand(8, 1)
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "mean_absolute_error",
                "mean_squared_error",
                "weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_list_losses(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=["mean_squared_error", "binary_crossentropy"],
            metrics=[
                "mean_squared_error",
                ["mean_squared_error", "accuracy"],
            ],
            loss_weights=[0.1, 2],
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_list_losses_abbr(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=["mse", "bce"],
            metrics=[
                ["bce", "mse", "mae"],
                ["mse", "acc"],
            ],
            loss_weights=[0.1, 2],
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_bce",
                "output_a_mae",
                "output_a_mse",
                "output_b_acc",
                "output_b_loss",
                "output_b_mse",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_nested_list_losses(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=["mean_squared_error", ["binary_crossentropy"]],
            metrics=[
                "mean_squared_error",
                ["mean_squared_error", "accuracy"],
            ],
            loss_weights=[0.1, 2],
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_dict_outputs_dict_losses(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": ["binary_crossentropy"],
            },
            metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
            weighted_metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs["output_a"].shape, (8, 1))
        self.assertEqual(outputs["output_b"].shape, (8, 1))
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            {"output_a": y1, "output_b": y2},
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
                "output_b_weighted_accuracy",
                "output_b_weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_dict_outputs_dict_losses_with_undefined_loss(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_b": ["binary_crossentropy"],
            },
            metrics={
                "output_b": ["mean_squared_error", "accuracy"],
            },
            weighted_metrics={
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(outputs["output_a"].shape, (8, 1))
        self.assertEqual(outputs["output_b"].shape, (8, 1))
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            {"output_a": y1, "output_b": y2},
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_b_accuracy",
                "output_b_mean_squared_error",
                "output_b_weighted_accuracy",
                "output_b_weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_metrics(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
            weighted_metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Check list outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs[0].shape, (8, 1))
        self.assertEqual(outputs[1].shape, (8, 1))
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
                "output_b_weighted_accuracy",
                "output_b_weighted_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_metrics_uniq_weighted(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["mean_squared_error"],
            },
            weighted_metrics={
                "output_a": ["mean_squared_error"],
                "output_b": ["accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        # `output_b_accuracy` doesn't have `weighted_` in metric name.
        # When a metric is only in weighted metrics, it skips `weighted_`
        # prefix. This behavior matches`tf.keras`.
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_a_mean_squared_error",
                "output_a_weighted_mean_squared_error",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_partial_metrics(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_b": ["mean_squared_error", "accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "output_a_loss",
                "output_b_accuracy",
                "output_b_loss",
                "output_b_mean_squared_error",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_dict_outputs_with_single_tensor(self):
        model = _get_model_multi_outputs_dict_with_single_tensor()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))

        # `model` has 2 outputs, but there is actually only 1 output tensor.
        self.assertLen(model.outputs, 2)
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
        )
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(["loss", "output_a_loss", "output_b_loss"])
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_with_custom_compute_loss(self):
        model = _get_model_with_custom_compute_loss()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))

        # `model` has 1 output, but in `compute_loss` it is separated into 2.
        self.assertLen(model.outputs, 1)
        model.compile(
            optimizer="sgd", loss=["mean_squared_error", "binary_crossentropy"]
        )
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            ["binary_crossentropy_loss", "loss", "mean_squared_error_loss"]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_list_outputs_dict_losses_invalid_keys(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_c": "binary_crossentropy",
            },
        )

        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "Expected keys",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_list_outputs_dict_losses_no_output_names(self):
        model = _get_model_multi_outputs_list_no_output_names()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={"output_a": "mean_squared_error"},
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "Expected keys",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_list_outputs_dict_metrics_invalid_keys(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_c": ["mean_squared_error", "accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "In the dict argument `metrics`, "
            "key 'output_c' does not correspond to any model output",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_dict_outputs_dict_losses_invalid_keys(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_c": "binary_crossentropy",
            },
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            KeyError,
            "in the `loss` argument, can't be found "
            "in either the model's output",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_dict_outputs_dict_metrics_invalid_keys(self):
        model = _get_model_multi_outputs_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss={
                "output_a": "mean_squared_error",
                "output_b": "binary_crossentropy",
            },
            metrics={
                "output_c": ["mean_squared_error", "accuracy"],
            },
        )
        # Fit the model to make sure compile_metrics are built
        with self.assertRaisesRegex(
            ValueError,
            "In the dict argument `metrics`, "
            "key 'output_c' does not correspond to any model output",
        ):
            model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)

    def test_functional_list_outputs_invalid_nested_list_losses(self):
        model = _get_model_multi_outputs_list()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.randint(0, 2, (8, 1))
        model.compile(
            optimizer="sgd",
            loss=[
                "mean_squared_error",
                ["mean_squared_error", "binary_crossentropy"],
            ],
        )
        hist = model.fit(x, (y1, y2), batch_size=2, epochs=1, verbose=0)
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(["loss", "output_a_loss", "output_b_loss"])
        self.assertListEqual(hist_keys, ref_keys)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
    )
    def test_quantize(self, mode):
        model = _get_model()
        x1 = np.random.rand(2, 3)
        x2 = np.random.rand(2, 3)
        model.quantize(mode)
        _ = model((x1, x2))

        for layer in model._flatten_layers():
            if isinstance(layer, (layers.Dense, layers.EinsumDense)):
                self.assertEqual(
                    layer.dtype_policy.name, f"{mode}_from_float32"
                )
                self.assertEqual(layer.dtype_policy.quantization_mode, mode)

    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
    )
    def test_quantize_unbuilt(self, mode):
        class MyModel(Model):
            def __init__(self):
                super().__init__()
                self.dense1 = layers.Dense(32, activation="relu")
                self.dense2 = layers.Dense(5, activation="softmax")
                self.dropout = layers.Dropout(0.5)

            def call(self, inputs, training=False):
                x = self.dense1(inputs)
                x = self.dropout(x, training=training)
                return self.dense2(x)

        model = MyModel()
        with self.assertRaisesRegex(
            ValueError, "Cannot quantize a layer that isn't yet built."
        ):
            model.quantize(mode)

        x = np.random.rand(2, 3)
        _ = model(x)
        model.quantize(mode)

    def test_quantize_invalid_args(self):
        model = _get_model()
        with self.assertRaisesRegex(
            ValueError, "Invalid quantization mode. Expected one of"
        ):
            model.quantize("abc")

        with self.assertRaisesRegex(
            ValueError, "Unrecognized keyword arguments"
        ):
            model.quantize("int8", unrecognized_kwargs=None)

        with self.assertRaisesRegex(ValueError, "Invalid quantization mode"):
            model.quantize("int7")
    
    @pytest.mark.slow
    def test_quantize_gptq_integration(self):
        """
        Tests that `model.quantize('gptq', ...)` correctly calls the backend.
        """
        # model_id = "facebook/opt-125m"
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # # Use the TF-specific class to get a TensorFlow/Keras-based model
        # model = TFAutoModelForCausalLM.from_pretrained(model_id)
        model = keras_nlp.models.OPTCausalLM.from_preset("opt_125m_en")
        # model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

        # --- CORRECTED PART ---
        # To access the transformer layers, we must first get the 'backbone' model.
        backbone = model.get_layer("opt_backbone_1")
        # backbone = model.get_layer("gemma_backbone")

        # Now we can get the specific transformer layer and its weights from the backbone.
        original_weights = np.copy(
            backbone.get_layer("transformer_layer_0")
            ._self_attention_layer._query_dense.kernel.numpy()
        )
        # original_weights = np.copy(
        #     backbone.get_layer("transformer_layer_0")
        #     .self_attention.query.kernel.numpy()
        # )
        long_text = """auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.
        The goal is to quantize pre-trained models to 4-bit or even 3-bit precision with minimal performance degradation.
        This allows for running larger models on less powerful hardware, reducing memory footprint and increasing inference speed.
        The process involves calibrating the model on a small dataset to determine the quantization parameters.
        This technique is particularly useful for deploying large language models in resource-constrained environments where every bit of memory and every millisecond of latency counts."""
        dataset = [long_text]
        # 2. Create the GPTQ configuration.
        gptq_config = GPTQConfig(
            dataset="wikitext2",
            # dataset=dataset,
            tokenizer=model.preprocessor.tokenizer,
            wbits=4,
            nsamples=128,
            seqlen=128,
            groupsize=128,
        )

        # 3. Run the actual quantization process by calling the config object directly.
        # This is the correct way to apply the logic to a third-party model object.
        # quantized_model = gptq_config.quantize(model)
        model.quantize("gptq", quant_config=gptq_config)

        quantized_weights = model.get_layer("opt_backbone_1").get_layer(
            "transformer_layer_0"
        )._self_attention_layer._query_dense.kernel.numpy()

        assert not np.allclose(
            original_weights, quantized_weights
        ), "The weights of the model were not changed by the quantization process."
    
        # dummy_input = tokenizer("Hello, world!", return_tensors="np")["input_ids"]
        dummy_input = ["Hello, world!"]
        try:
            _ = model.predict(dummy_input)
        except Exception as e:
            pytest.fail(f"The quantized model failed during predict(): {e}")



    @parameterized.named_parameters(
        ("int8", "int8"),
        ("float8", "float8"),
    )
    def test_quantize_nested_model(self, mode):
        class NestedLayer(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.dense = layers.Dense(units)

            def call(self, x):
                x = self.dense(x)
                return x

        class DoubleNestedLayer(layers.Layer):
            def __init__(self, units):
                super().__init__()
                self.nested_dense1 = NestedLayer(units)
                self.nested_dense2 = NestedLayer(units)
                self.dense = layers.Dense(units)

            def call(self, x):
                x = self.nested_dense1(x)
                x = self.nested_dense2(x)
                x = self.dense(x)
                return x

        inputs = layers.Input([3])
        outputs = DoubleNestedLayer(8)(inputs)
        model = Model(inputs, outputs)
        model.quantize(mode)

        if mode == "int8":
            kernel_count = 0
            for weight in model.weights:
                if weight.name == "kernel":
                    kernel_count += 1
                    self.assertEqual(
                        backend.standardize_dtype(weight.dtype), "int8"
                    )
            self.assertEqual(kernel_count, 3)
        if mode == "float8":
            # kernel + bias + scale * 3 + amax_history * 3 == 8
            self.assertEqual(len(model.weights), 3 * 8)

    def test_get_state_tree(self):
        model = _get_model_single_output()
        model.compile(loss="mse", optimizer="adam")
        state_tree = model.get_state_tree()
        self.assertAllClose(
            state_tree["trainable_variables"]["output_a"]["kernel"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/kernel"
            ),
        )
        self.assertAllClose(
            state_tree["trainable_variables"]["output_a"]["bias"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/bias"
            ),
        )
        self.assertEqual(
            state_tree["non_trainable_variables"],
            {},
        )
        self.assertEqual(
            state_tree["metrics_variables"]["loss"]["count"],
            _get_variable_value_by_path(model.metrics_variables, "loss/count"),
        )
        self.assertEqual(
            state_tree["metrics_variables"]["loss"]["total"],
            _get_variable_value_by_path(model.metrics_variables, "loss/total"),
        )
        self.assertEqual(
            state_tree["optimizer_variables"]["adam"]["iteration"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/iteration"
            ),
        )
        self.assertEqual(
            state_tree["optimizer_variables"]["adam"]["learning_rate"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/learning_rate"
            ),
        )

        # Test with numpy
        state_tree = model.get_state_tree(value_format="numpy_array")
        self.assertIsInstance(
            state_tree["trainable_variables"]["output_a"]["kernel"], np.ndarray
        )

    def test_set_state_tree(self):
        variables = {
            "optimizer_variables": {
                "adam": {
                    "iteration": 0,
                    "learning_rate": 0.00001,
                }
            },
            "trainable_variables": {
                "output_a": {
                    "bias": [0.5],
                    "kernel": [[0.6], [0.7], [1.8]],
                }
            },
        }

        model = _get_model_single_output()
        model.compile(optimizer="adam")
        model.set_state_tree(variables)

        self.assertEqual(
            variables["optimizer_variables"]["adam"]["iteration"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/iteration"
            ),
        )
        self.assertEqual(
            variables["optimizer_variables"]["adam"]["learning_rate"],
            _get_variable_value_by_path(
                model.optimizer.variables, "adam/learning_rate"
            ),
        )
        self.assertAllClose(
            variables["trainable_variables"]["output_a"]["bias"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/bias"
            ),
        )
        self.assertAllClose(
            variables["trainable_variables"]["output_a"]["kernel"],
            _get_variable_value_by_path(
                model.trainable_variables, "output_a/kernel"
            ),
        )

    def test_get_state_tree_with_duplicate_path(self):
        model = _get_model_with_duplicate_variable_path()
        with self.assertRaisesRegex(
            ValueError,
            "The following variable path is found twice in the model",
        ):
            model.get_state_tree()

    def test_layers_setter(self):
        model = Model()
        with self.assertRaisesRegex(
            AttributeError, "`Model.layers` attribute is reserved"
        ):
            model.layers = [layers.Dense(4)]

    def get_struct_loss(self, structure):
        def loss_fn(y_true, y_pred):
            tree.assert_same_structure(structure, y_true)
            tree.assert_same_structure(structure, y_pred)
            tree.map_structure(
                lambda spec, tensor: self.assertEqual(spec.ndim, tensor.ndim),
                structure,
                y_true,
            )
            tree.map_structure(
                lambda spec, tensor: self.assertEqual(spec.ndim, tensor.ndim),
                structure,
                y_pred,
            )
            flat_y_pred = tree.flatten(y_pred)
            flat_y_true = tree.flatten(y_true)
            diff = 0
            for y_p, y_t in zip(flat_y_pred, flat_y_true):
                diff += losses.mean_absolute_error(y_t, y_p)
            return diff

        return loss_fn

    @parameterized.product(
        _type=[tuple, list], other_type=[list, tuple], weighted=[False, True]
    )
    def test_functional_struct_outputs_struct_losses(
        self, _type, other_type, weighted
    ):
        model = _get_model_multi_outputs_struct_list_like(_type)
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)
        y = _type([y1, y2])
        loss = other_type(
            [
                self.get_struct_loss(model.output),
                _type(
                    [
                        self.get_struct_loss(model.output[0]),
                        self.get_struct_loss(model.output[1]),
                    ]
                ),
            ]
        )
        if weighted:
            loss_weights = tree.map_structure(lambda _: np.random.rand(), loss)
        else:
            loss_weights = None

        model.compile(
            optimizer="sgd",
            loss=loss,
            loss_weights=loss_weights,
        )

        if _type is other_type:
            with self.assertRaisesRegex(
                ValueError, "[Ee]xpected.*" + _type.__name__
            ):
                model.fit(x, y, batch_size=2, epochs=1, verbose=0)
        else:
            # Check dict outputs.
            outputs = model.predict(x)
            self.assertIsInstance(outputs, _type)
            # Fit the model to make sure compile_metrics are built
            hist = model.fit(
                x,
                y,
                batch_size=2,
                epochs=1,
                verbose=0,
            )
            hist_keys = sorted(hist.history.keys())
            ref_keys = sorted(
                [
                    "loss",
                    "y1_loss",
                    "y2_loss",
                    "y1_y2_loss",
                ]
            )
            self.assertListEqual(hist_keys, ref_keys)

    @parameterized.named_parameters(("weighted", True), ("not_weighted", False))
    def test_functional_struct_outputs_dict_struct_losses(self, weighted):
        model = _get_model_multi_outputs_struct_dict()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)

        y = {"a": y1, "b": y2}
        loss = [
            self.get_struct_loss(model.output),
            {
                "a": self.get_struct_loss(model.output["a"]),
                "b": self.get_struct_loss(model.output["a"]),
            },
        ]
        if weighted:
            loss_weights = tree.map_structure(lambda _: np.random.rand(), loss)
        else:
            loss_weights = None

        model.compile(
            optimizer="sgd",
            loss=loss,
            loss_weights=loss_weights,
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)

        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "a_loss",
                "b_loss",
                "a_b_loss",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_struct_outputs_namedtuple_struct_losses(self):
        model, Y = _get_model_multi_outputs_struct_namedtuple()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)

        y = Y(y1, y2)
        model.compile(
            optimizer="sgd",
            loss=[
                self.get_struct_loss(model.output),
                Y(
                    self.get_struct_loss(model.output.y1),
                    self.get_struct_loss(model.output.y2),
                ),
            ],
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, tuple)

        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "loss",
                "y1_loss",
                "y2_loss",
                "y1_y2_loss",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_functional_deeply_nested_outputs_struct_losses(self):
        model = _get_model_multi_outputs_struct()
        self.assertIsInstance(model, Functional)
        x = np.random.rand(8, 3)
        y1 = np.random.rand(8, 1)
        y2 = np.random.rand(8, 1)
        y3 = np.random.rand(8, 1)
        y = {
            "a": (y1, y2),
            "b": {"b1": y1, "b2": y2},
            "c": {"c1": (y1, y2), "c2": y2},
            "d": y3,
        }
        model.compile(
            optimizer="sgd",
            loss={
                "a": [
                    self.get_struct_loss(model.output["a"]),
                    (None, self.get_struct_loss(model.output["a"][1])),
                ],
                "b": [
                    self.get_struct_loss(model.output["b"]),
                    {"b1": self.get_struct_loss(model.output["b"]["b1"])},
                ],
                "c": [
                    self.get_struct_loss(model.output["c"]),
                    {"c1": self.get_struct_loss(model.output["c"]["c1"])},
                ],
                "d": self.get_struct_loss(model.output["d"]),
            },
        )
        # Check dict outputs.
        outputs = model.predict(x)
        self.assertIsInstance(outputs, dict)

        # Fit the model to make sure compile_metrics are built
        hist = model.fit(
            x,
            y,
            batch_size=2,
            epochs=1,
            verbose=0,
        )
        hist_keys = sorted(hist.history.keys())
        ref_keys = sorted(
            [
                "a/y2_loss",
                "a_loss",
                "b/b1_loss",
                "b_loss",
                "c/c1_loss",
                "c_loss",
                "d_loss",
                "loss",
            ]
        )
        self.assertListEqual(hist_keys, ref_keys)

    def test_export_error(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "exported_model")
        model = _get_model()

        # Bad format
        with self.assertRaisesRegex(ValueError, "Unrecognized format="):
            model.export(temp_filepath, format="bad_format")

        # Bad backend
        if backend.backend() not in ("tensorflow", "jax", "torch"):
            with self.assertRaisesRegex(
                NotImplementedError,
                (
                    r"`export_saved_model` only currently supports the "
                    r"tensorflow, jax and torch backends."
                ),
            ):
                model.export(temp_filepath, format="tf_saved_model")



import keras
import keras_nlp
import numpy as np
import pytest
from keras.src import layers, models, testing
from keras.src.quantizers.gptqconfig import GPTQConfig

# Helper function to generate dummy data for quick testing.
def dummy_dataset_generator(nsamples, seqlen, vocab_size=1000):
    """A generator that yields random numpy arrays for fast, self-contained tests."""
    for _ in range(nsamples):
        yield np.random.randint(0, vocab_size, size=(1, seqlen))

# Helper function to build a simple transformer model that uses standard 
# Keras `Dense` layers for its attention projections.
def _get_model_with_dense_attention():
    """Builds a simple transformer model using Dense for attention."""
    vocab_size = 1000
    embed_dim = 32
    num_heads = 4
    ff_dim = 32
    seq_len = 128

    class SimpleTransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
            super().__init__(**kwargs)
            # The standard MultiHeadAttention layer uses Dense layers for its projections.
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = models.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs):
            attention_output = self.att(inputs, inputs)
            out1 = self.layernorm1(inputs + attention_output)
            ffn_output = self.ffn(out1)
            return self.layernorm2(out1 + ffn_output)

    inputs = layers.Input(shape=(None,), dtype="int32")
    embedding_layer = layers.Embedding(vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = SimpleTransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

@pytest.mark.requires_trainable_backend
class ModelQuantizationTest(testing.TestCase):
    
    @pytest.mark.slow
    def test_quantize_gptq_with_dense_attention(self):
        """Tests GPTQ on a model with Dense layers in its attention block."""
        model = _get_model_with_dense_attention()
        
        # Create a mock tokenizer for the config, as the model is simple.
        long_text = """auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.
        The goal is to quantize pre-trained models to 4-bit or even 3-bit precision with minimal performance degradation.
        This allows for running larger models on less powerful hardware, reducing memory footprint and increasing inference speed.
        The process involves calibrating the model on a small dataset to determine the quantization parameters.
        This technique is particularly useful for deploying large language models in resource-constrained environments where every bit of memory and every millisecond of latency counts."""
        dataset = [long_text]

        mock_tokenizer = lambda text: np.array([ord(c) for c in text])
        mock_tokenizer.tokenize = mock_tokenizer

        # Get original weights from a dense layer to compare against after quantization.
        original_weights = np.copy(model.layers[2].ffn.layers[0].kernel.numpy())

        # Configure the GPTQ quantizer with dummy data.
        gptq_config = GPTQConfig(
            dataset=dummy_dataset_generator(nsamples=16, seqlen=128, vocab_size=1000),
            tokenizer=mock_tokenizer,
            wbits=4,
            nsamples=16,  # Use a small number of samples for a fast test
            seqlen=128,
            groupsize=32, # Use a smaller group size for the smaller layers
        )

        # Run the quantization process.
        model.quantize("gptq", quant_config=gptq_config)

        # Get the new weights after quantization.
        quantized_weights = model.layers[2].ffn.layers[0].kernel.numpy()

        # 1. Assert that the weights have actually been changed by the process.
        self.assertFalse(
            np.allclose(original_weights, quantized_weights),
            "Weights were not changed by the GPTQ process for the Dense attention model."
        )
        
        # 2. Verify the quantized model can still make a prediction without crashing.
        try:
            dummy_input = np.random.randint(0, 1000, size=(1, 128))
            _ = model.predict(dummy_input)
        except Exception as e:
            self.fail(f"Prediction failed for the quantized Dense attention model: {e}")


    @pytest.mark.slow
    def test_quantize_gptq_with_gpt2(self):
        from datasets import load_dataset
        model = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en_cnn_dailymail")
        # model = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")

        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # Prepare the data for the perplexity function
        # This joins the text and tokenizes it, similar to your get_dataloader function
        all_text = "\n\n".join(d['text'] for d in test_data if d['text'])
        all_tokens = model.preprocessor.tokenizer.tokenize(all_text)

        # Create a few samples from the real test data
        test_samples = []
        seq_len = 128
        for i in range(50): # Use 50 samples for a stable PPL score
            start = i * seq_len
            end = start + seq_len
            test_samples.append(np.reshape(all_tokens[start:end], (1, seq_len)))

        test_dataloader = np.array(test_samples, dtype=np.int32)
        long_text = """auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm.
        The goal is to quantize pre-trained models to 4-bit or even 3-bit precision with minimal performance degradation.
        This allows for running larger models on less powerful hardware, reducing memory footprint and increasing inference speed.
        The process involves calibrating the model on a small dataset to determine the quantization parameters.
        This technique is particularly useful for deploying large language models in resource-constrained environments where every bit of memory and every millisecond of latency counts."""
        dataset = [long_text]
        # 2. Create the GPTQ configuration.
        gptq_config = GPTQConfig(
            dataset="wikitext2",
            # dataset=dataset,
            tokenizer=model.preprocessor.tokenizer,
            wbits=4,
            nsamples=128,
            seqlen=128,
            groupsize=128,
        )

        # 3. Run the actual quantization process by calling the config object directly.
        # This is the correct way to apply the logic to a third-party model object.
        # quantized_model = gptq_config.quantize(model)
        model.quantize("gptq", quant_config=gptq_config)

        test_dataloader = np.array(test_samples, dtype=np.int32)
        perplexity = calculate_perplexity(model, test_dataloader)
        self.assertLess(perplexity, 200, "Perplexity should be low for a pre-trained model on real data.")


    @pytest.mark.slow
    def test_quantize_gptq_with_einsumdense_attention(self):
        from datasets import load_dataset
        import tempfile
        import os

        model = keras_nlp.models.Gemma3CausalLM.from_preset("gemma3_1b", load_weights=True)
        # Load the test split of the wikitext2 dataset
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # Prepare the data for the perplexity function
        # This joins the text and tokenizes it, similar to your get_dataloader function
        all_text = "\n\n".join(d['text'] for d in test_data if d['text'])
        all_tokens = model.preprocessor.tokenizer.tokenize(all_text)

        # Create a few samples from the real test data
        test_samples = []
        seq_len = 128
        for i in range(50): # Use 50 samples for a stable PPL score
            start = i * seq_len
            end = start + seq_len
            test_samples.append(np.reshape(all_tokens[start:end], (1, seq_len)))

        test_dataloader = np.array(test_samples, dtype=np.int32)
        perplexity = calculate_perplexity(model, test_dataloader)
        self.assertLess(perplexity, 200, "Perplexity should be low for a pre-trained model on real data.")


        gptq_config = GPTQConfig(
            # dataset=dummy_dataset_generator(nsamples=16, seqlen=128, vocab_size=model.preprocessor.tokenizer.vocabulary_size()),
            dataset="wikitext2",
            tokenizer=model.preprocessor.tokenizer,
            wbits=4,
            nsamples=128,
            seqlen=128,
            groupsize=128,
        )
        model.quantize("gptq", quant_config=gptq_config)

                # 1. Save the newly quantized weights to a temporary file.
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_path = os.path.join(tmpdir, "quantized_weights.weights.h5")
            model.save_weights(weights_path)

            # 2. Create a NEW, clean instance of the model. This instance has a
            #    perfectly functional tokenizer and a clean internal state.
            clean_model = keras_nlp.models.Gemma3CausalLM.from_preset("gemma3_1b", load_weights=False)

            # 3. Load the quantized weights into the new model.
            clean_model.load_weights(weights_path)
        
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # Prepare the data for the perplexity function
        # This joins the text and tokenizes it, similar to your get_dataloader function
        all_text = "\n\n".join(d['text'] for d in test_data if d['text'])
        all_tokens = clean_model.preprocessor.tokenizer.tokenize(all_text)

        # Create a few samples from the real test data
        test_samples = []
        seq_len = 128
        for i in range(50): # Use 50 samples for a stable PPL score
            start = i * seq_len
            end = start + seq_len
            test_samples.append(np.reshape(all_tokens[start:end], (1, seq_len)))

        test_dataloader = np.array(test_samples, dtype=np.int32)
        perplexity = calculate_perplexity(clean_model, test_dataloader)
        self.assertLess(perplexity, 200, "Perplexity should be low for a pre-trained model on real data.")