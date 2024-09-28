import pytest
import jax
from jax import numpy as jnp
import numpy as np
from pupperv3_mjx.utils import activation_fn_map, circular_buffer_shift_back


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


def test_circular_buffer_push_back():
    buffer = jnp.array([[1, 2, 3], [4, 5, 6]])
    new_value = jnp.array([7, 8])
    expected_output = jnp.array([[2, 3, 7], [5, 6, 8]])
    output = circular_buffer_push_back(buffer, new_value)
    np.testing.assert_array_equal(output, expected_output)
