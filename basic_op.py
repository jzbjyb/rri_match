from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_functional_ops import *
# pylint: enable=wildcard-import
# pylint: disable=unused-import
from tensorflow.python.ops.gen_functional_ops import _symbolic_gradient
# pylint: enable=unused-import
from tensorflow.python.util import nest

def map_fn_concat(fn, elems, dtype=None, parallel_iterations=10, back_prop=True,
           swap_memory=False, infer_shape=True, name=None):
  """map on the list of tensors unpacked from `elems` on dimension 0.

  The simplest version of `map_fn` repeatedly applies the callable `fn` to a
  sequence of elements from first to last. The elements are made of the
  tensors unpacked from `elems`. `dtype` is the data type of the return
  value of `fn`. Users must provide `dtype` if it is different from
  the data type of `elems`.

  Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
  of the result tensor is `[values.shape[0]] + fn(values[0]).shape`.

  This method also allows multi-arity `elems` and output of `fn`.  If `elems`
  is a (possibly nested) list or tuple of tensors, then each of these tensors
  must have a matching first (unpack) dimension.  The signature of `fn` may
  match the structure of `elems`.  That is, if `elems` is
  `(t1, [t2, t3, [t4, t5]])`, then an appropriate signature for `fn` is:
  `fn = lambda (t1, [t2, t3, [t4, t5]]):`.

  Furthermore, `fn` may emit a different structure than its input.  For example,
  `fn` may look like: `fn = lambda t1: return (t1 + 1, t1 - 1)`.  In this case,
  the `dtype` parameter is not optional: `dtype` must be a type or (possibly
  nested) tuple of types matching the output of `fn`.

  To apply a functional operation to the nonzero elements of a SparseTensor
  one of the following methods is recommended. First, if the function is
  expressible as TensorFlow ops, use

  ```python
    result = SparseTensor(input.indices, fn(input.values), input.dense_shape)
  ```

  If, however, the function is not expressible as a TensorFlow op, then use

  ```python
  result = SparseTensor(
    input.indices, map_fn(fn, input.values), input.dense_shape)
  ```

  instead.

  Args:
    fn: The callable to be performed.  It accepts one argument, which will
      have the same (possibly nested) structure as `elems`.  Its output
      must have the same structure as `dtype` if one is provided, otherwise
      it must have the same structure as `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which
      will be unpacked along their first dimension.  The nested sequence
      of the resulting slices will be applied to `fn`.
    dtype: (optional) The output type(s) of `fn`.  If `fn` returns a structure
      of Tensors differing from the structure of `elems`, then `dtype` is not
      optional and must have the same structure as the output of `fn`.
    parallel_iterations: (optional) The number of iterations allowed to run
      in parallel.
    back_prop: (optional) True enables support for back propagation.
    swap_memory: (optional) True enables GPU-CPU memory swapping.
    infer_shape: (optional) False disables tests for consistent output shapes.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor or (possibly nested) sequence of tensors.  Each tensor packs the
    results of applying `fn` to tensors unpacked from `elems` along the first
    dimension, from first to last.

  Raises:
    TypeError: if `fn` is not callable or the structure of the output of
      `fn` and `dtype` do not match, or if elems is a SparseTensor.
    ValueError: if the lengths of the output of `fn` and `dtype` do not match.

  Examples:
    ```python
    elems = np.array([1, 2, 3, 4, 5, 6])
    squares = map_fn(lambda x: x * x, elems)
    # squares == [1, 4, 9, 16, 25, 36]
    ```

    ```python
    elems = (np.array([1, 2, 3]), np.array([-1, 1, -1]))
    alternate = map_fn(lambda x: x[0] * x[1], elems, dtype=tf.int64)
    # alternate == [-1, 2, -3]
    ```

    ```python
    elems = np.array([1, 2, 3])
    alternates = map_fn(lambda x: (x, -x), elems, dtype=(tf.int64, tf.int64))
    # alternates[0] == [1, 2, 3]
    # alternates[1] == [-1, -2, -3]
    ```
  """
  if not callable(fn):
    raise TypeError("fn must be callable.")

  if isinstance(elems, sparse_tensor.SparseTensor):
    raise TypeError(
        "To perform a map on the values of a sparse tensor use either "
        " SparseTensor(input.indices, fn(input.values), input.dense_shape) or "
        " SparseTensor(input.indices, map_fn(fn, input.values), "
        "input.dense_shape)")

  input_is_sequence = nest.is_sequence(elems)
  input_flatten = lambda x: nest.flatten(x) if input_is_sequence else [x]
  def input_pack(x):
    return nest.pack_sequence_as(elems, x) if input_is_sequence else x[0]

  if dtype is None:
    output_is_sequence = input_is_sequence
    output_flatten = input_flatten
    output_pack = input_pack
  else:
    output_is_sequence = nest.is_sequence(dtype)
    output_flatten = lambda x: nest.flatten(x) if output_is_sequence else [x]
    def output_pack(x):
      return (nest.pack_sequence_as(dtype, x)
              if output_is_sequence else x[0])

  elems_flat = input_flatten(elems)

  with ops.name_scope(name, "map", elems_flat):
    # Any get_variable calls in fn will cache the first call locally
    # and not issue repeated network I/O requests for each iteration.
    varscope = vs.get_variable_scope()
    varscope_caching_device_was_none = False
    if varscope.caching_device is None:
      # TODO(ebrevdo): Change to using colocate_with here and in other methods.
      varscope.set_caching_device(lambda op: op.device)
      varscope_caching_device_was_none = True

    elems_flat = [
        ops.convert_to_tensor(elem, name="elem") for elem in elems_flat]

    dtype = dtype or input_pack([elem.dtype for elem in elems_flat])
    dtype_flat = output_flatten(dtype)

    # Convert elems to tensor array.
    n = array_ops.shape(elems_flat[0])[0]

    # TensorArrays are always flat
    elems_ta = [
        tensor_array_ops.TensorArray(dtype=elem.dtype, size=n,
                                     dynamic_size=False,
                                     infer_shape=True)
        for elem in elems_flat]
    # Unpack elements
    elems_ta = [
        elem_ta.unstack(elem) for elem_ta, elem in zip(elems_ta, elems_flat)]

    i = constant_op.constant(0)

    accs_ta = [
        tensor_array_ops.TensorArray(dtype=dt, size=n,
                                     dynamic_size=False,
                                     infer_shape=infer_shape)
        for dt in dtype_flat]

    def compute(i, tas):
      """The loop body of map_fn.

      Args:
        i: the loop counter
        tas: the flat TensorArray accumulator list

      Returns:
        (i + 1, tas): the updated counter + updated TensorArrays

      Raises:
        TypeError: if dtype and packed_fn_values structure do not match
        ValueType: if dtype and packed_fn_values lengths do not match
      """
      packed_values = input_pack([elem_ta.read(i) for elem_ta in elems_ta])
      packed_fn_values = fn(packed_values)
      nest.assert_same_structure(dtype or elems, packed_fn_values)
      flat_fn_values = output_flatten(packed_fn_values)
      tas = [ta.write(i, value) for (ta, value) in zip(tas, flat_fn_values)]
      return (i + 1, tas)

    _, r_a = control_flow_ops.while_loop(
        lambda i, _: i < n, compute, (i, accs_ta),
        parallel_iterations=parallel_iterations,
        back_prop=back_prop,
        swap_memory=swap_memory)
    results_flat = [r.concat() for r in r_a]

    n_static = elems_flat[0].get_shape().with_rank_at_least(1)[0]
    for elem in elems_flat[1:]:
      n_static.merge_with(elem.get_shape().with_rank_at_least(1)[0])
    for r in results_flat:
      r.set_shape(tensor_shape.TensorShape(n_static).concatenate(
          r.get_shape()[1:]))

    if varscope_caching_device_was_none:
      varscope.set_caching_device(None)

    return output_pack(results_flat)