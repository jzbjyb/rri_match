import contextlib
import numpy as np
import tensorflow as tf


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def tf_jacobian(y_flat, x):
    n = y_flat.shape[1]
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[:, j], x)[0])),
        loop_vars)
    jacobian = jacobian.stack()
    x_len = len(x.get_shape())
    jacobian = tf.transpose(jacobian, [1, 0] + list(range(2, x_len + 1)))
    return jacobian