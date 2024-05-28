import pytest
import jax
from jax import numpy as jnp
import numpy as np
from pupperv3_mjx.utils import activation_fn_map


def test_relu():
    fn = activation_fn_map("relu")
    input_val = jnp.array([-1.0, 0.0, 1.0])
    expected_output = jnp.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(fn(input_val), expected_output)


def test_sigmoid():
    fn = activation_fn_map("sigmoid")
    input_val = jnp.array([-1.0, 0.0, 1.0])
    expected_output = 1 / (1 + jnp.exp(-input_val))
    np.testing.assert_array_almost_equal(fn(input_val), expected_output)


def test_elu():
    fn = activation_fn_map("elu")
    input_val = jnp.array([-1.0, 0.0, 1.0])
    expected_output = jax.nn.elu(input_val)
    np.testing.assert_array_almost_equal(fn(input_val), expected_output)


def test_tanh():
    fn = activation_fn_map("tanh")
    input_val = jnp.array([-1.0, 0.0, 1.0])
    expected_output = jnp.tanh(input_val)
    np.testing.assert_array_almost_equal(fn(input_val), expected_output)


def test_softmax():
    fn = activation_fn_map("softmax")
    input_val = jnp.array([1.0, 2.0, 3.0])
    expected_output = jax.nn.softmax(input_val)
    np.testing.assert_array_almost_equal(fn(input_val), expected_output)


def test_invalid_activation():
    with pytest.raises(KeyError):
        activation_fn_map("invalid")
