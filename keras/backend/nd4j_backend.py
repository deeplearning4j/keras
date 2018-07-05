from keras.backend.common import floatx, epsilon
from keras.backend.common import image_data_format

from jumpy.java_classes import SameDiff, Nd4j, Transforms
from jumpy.ndarray import array, get_context_dtype, set_context_dtype
from jumpy.matlib import zeros as nd4j_zeros
from jumpy.matlib import ones as nd4j_ones
from .graph import Placeholder, Variable, Op, Graph



from collections import defaultdict


import numpy as np   # TODO : remove
from jumpy.java_classes import *
from jumpy.ndarray import array, ndarray


_INDArray_class = 'org.nd4j.linalg.api.ndarray.INDArray'
_SD_class = 'org.nd4j.autodiff.samediff.SDVariable'


def is_numpy(x):
    return 'numpy' in x.__class__.__name__


def is_nd4j(x):
    return type(x).__name__ == _INDArray_class


def is_jumpy(x):
    return type(x) == ndarray

def is_sd(x):
    return type(x).__name__ == _SD_class


def is_tensor(x):
    return isinstance(x, Placeholder)


'''
Use the @op decorator over a method to automatically
take care of nd4j<->jumpy conversions. e.g:

```python

@op
def reshape(arr, shape):
    # we are in nd4j space now
    # arr is an INDArray instance
    # we return a INDArray instance as well
    return arr.reshape(*shape)


# use in jumpy space:

x = jp.zeros((2, 2, 3))  # x is jumpy ndarray
y = reshape(x, (4, 3))  # y is a jumpy ndarray

```

Note that methods with first argument named 'arr'
will be automatically bound to ndarray class.

'''
def op(f):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if is_jumpy(arg):
                args[i] = arg.array
        for k in kwargs:
            v = kwargs[k]
            if is_jumpy(v):
                kwargs[k] = v.array
        out = f(*args, **kwargs)
        if is_nd4j(out):
            return array(out)
        elif type(out) is list:
            for i, v in enumerate(out):
                if is_nd4j(v):
                    out[i] = array(v)
            return out
        elif type(out) is tuple:
            out = list(out)
            for i, v in enumerate(out):
                if is_nd4j(v):
                    out[i] = array(v)
            return tuple(out)
    return wrapper

sd = SameDiff.create()
_varid = -1
def _varname():
    _varid += 1
    return "var" + str(_varid)
def sdvar(x):
    return sd.var(_varname(), x)


# Note: sdops are super slow.. don't misuse;
def sdop(f):
    def wrapper(*args, **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if is_jumpy(arg):
                args[i] = sdvar(arg.array)
        for k in kwargs:
            v = kwargs[k]
            if is_jumpy(v):
                kwargs[k] = sdvar(v.array)
        
        out = f(*args, **kwargs)
        if is_nd4j(out):
            return array(out)
        elif is_sd(out):
            return array(out.eval())
        elif type(out) is list:
            for i, v in enumerate(out):
                if is_nd4j(v):
                    out[i] = array(v)
                elif is_sd(v):
                    out[i] = array(v.eval())
            return out
        elif type(out) is tuple:
            out = list(out)
            for i, v in enumerate(out):
                if is_nd4j(v):
                    out[i] = array(v)
                elif is_sd(v):
                    out[i] = array(v.eval())
            return tuple(out)
    return wrapper

def _to_int_shape(x):
    if x is None:
        return -1
    else:
        return [-1 if i is None else i for i in x]

opnames = set()
def get_opname(op):
    i = 0
    while(True):
        opname = op + '_' + str(i)
        if opname not in opnames:
            opnames.add(opname)
            return opname
        i += 1

uids = defaultdict(lambda *_: 0)
def get_uid(prefix=''):
    uid = uids[prefix]
    uids[prefix] += 1
    return uid


# TODO
class name_scope(object):

    def __init__(self, name):
        self.name = name

    def __enter__(self, *args, **kwargs):
        pass
    def __exit__(self, *args, **kwargs):
        pass



def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    if shape is None:
        if ndim is None:
            shape = None
        else:
            shape = [None for _ in range(ndim)]
    ph = Placeholder(name=name, _keras_shape=shape, _uses_learning_phase=False)
    return ph


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        minval: A float, lower boundary of the uniform distribution
            to draw samples.
        maxval: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if seed is None:
        seed = np.random.randint(10e6)
    return array(np.random.uniform(minval, maxval, shape))


def is_placeholder(x):
    return getattr(x, 'placeholder', False)


def variable(value, dtype=None, name=None, constraint=None):
    return array(value)


def constant(value, dtype=None, shape=None, name=None):
    if type(value) in (int, float):
        if shape is None:
            shape = []
        c = nd4j_zeros(shape)
        c += value
        return c


def is_keras_tensor(x):
    return hasattr(x, '_keras_history')

@op
def shape(x):
    return x.shape()


def int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    if is_nd4j(x):
        return tuple(x.shape())
    if is_sd(x):
        return tuple(x.add(0).eval().getShape())
    if is_jumpy(x):
        return x.shape


def ndim(x):
    shape = int_shape(x)
    if shape is None:
        return shape
    return len(shape)


def dtype(x):
    return get_context_dtype()

@op
def eval(x):
    return x


def zeros(shape, dtype=None, name=None):
    return nd4j_zeros(shape)


def ones(shape, dtype=None, name=None):
    return nd4j_ones(shape)


def eye(size, dtype=None, name=None):
    return Nd4j.eye(size)

@op
def zeros_like(x, dtype=None, name=None):
    return Nd4j.zerosLike(x)

@op
def ones_like(x, dtype=None, name=None):
    return Nd4j.onesLike(x)

@op
def identity(x,name=None):
    return x.dup()


def random_uniform_variable(shape, low, high, dtype=None,
                            name=None, seed=None):
    raise NotImplementedError


def random_normal_variable(shape, mean, scale, dtype=None,
                           name=None, seed=None):
    raise NotImplementedError

def cast(x, dtype):
    raise NotImplementedError


@op
def count_params(x):
    p = 1
    s = int_shape(x)
    for i in s:
        p *= i
    return p

@op
def update(x, new_x):
    """Update the value of `x` to `new_x`.

    # Arguments
        x: A `Variable`.
        new_x: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return x.assign(new_x)

@op
def update_add(x, increment):
    """Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return x.addi(increment)

@op
def update_sub(x, decrement):
    """Update the value of `x` by subtracting `decrement`.

    # Arguments
        x: A `Variable`.
        decrement: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return x.subi(decrement)


def moving_average_update(x, value, momentum):
    raise NotImplementedError

@op
def dot(x, y):
    x_shape = x.shape()
    if len(x_shape) > 2:
        batch_dim = 1
        for d in x_shape[:-1]:
            batch_dim *= d
        x = x.reshape(batch_dim, x_shape[-1])
        z = x.mmul(y)
        z = z.reshape(*(x_shape[:-1] + [z.shape()[-1]]))
        return z
    return x.mmul(y)

@op
def batch_dot(x, y, axes=None):
    if isinstance(axes, int):
        axes = [[axes], [axes]]
    if axes is None:
        # behaves like tf.batch_matmul as default
        x_ndim = ndim(x)
        y_ndim = ndim(y)
        axes = [[x_ndim - 1], [y_ndim - 2]]
    axes = [[a] if type(a) is int else a for a in axes]
    return Nd4j.tensorMmul(x, y, axes)

@op
def transpose(x):
    return x.transpose()




@sdop
def gather(reference, indices):
    """Retrieves the elements of indices `indices` in the tensor `reference`.

    # Arguments
        reference: A tensor
        indices: An integer tensor of indices.

    # Returns
        A tensor of same type as `reference`.
    """
    y = sd.gatherNd(reference, indices)
    return y

# ELEMENT-WISE OPERATIONS

@op
def max(x, axis=None, keepdims=False):
    """Maximum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find maximum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with maximum values of `x`.
    """
    if axis is None:
        if keepdims:
            return Nd4j.max(x).reshape(* [1] * ndim(x))
        else:
            return Nd4j.max(x)
    mx = Nd4j.max(x, axis)
    if keepdims:
        mx = Nd4j.expandDims(mx, axis)
    return mx

@op
def min(x, axis=None, keepdims=False):
    """Minimum value in a tensor.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to find minimum values.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with miminum values of `x`.
    """
    if axis is None:
        if keepdims:
            return Nd4j.min(x).reshape(* [1] * ndim(x))
        else:
            return Nd4j.min(x)
    mn = Nd4j.min(x, axis)
    if keepdims:
        mn = Nd4j.expandDims(mn, axis)
    return mn

@op
def sum(x, axis=None, keepdims=False):
    """Sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to sum over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with sum of `x`.
    """
    if axis is None:
        if keepdims:
            return Nd4j.sum(x).reshape(* [1] * ndim(x))
        else:
            return Nd4j.sum(x)
    s = Nd4j.sum(x, axis)
    if keepdims:
        s = Nd4j.expandDims(s, axis)
    return s

@op
def prod(x, axis=None, keepdims=False):
    """Multiplies the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the product of elements of `x`.
    """
    if axis is None:
        if keepdims:
            return Nd4j.prod(x).reshape(* [1] * ndim(x))
        else:
            return Nd4j.prod(x)
    p = Nd4j.prod(x, axis)
    if keepdims:
        p = Nd4j.expandDims(s, axis)
    return p


@op
def cumsum(x, axis=0):
    """Cumulative sum of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the sum.

    # Returns
        A tensor of the cumulative sum of values of `x` along `axis`.
    """
    return Nd4j.cumsum(x, axis)

@op
def cumprod(x, axis=0):
    """Cumulative product of the values in a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the product.

    # Returns
        A tensor of the cumulative product of values of `x` along `axis`.
    """
    return Nd4j.cumprod(x, axis)

@op
def var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    if axis is None:
        if keepdims:
            return Nd4j.var(x).reshape(* [1] * ndim(x))
        else:
            return Nd4j.var(x)
    v = Nd4j.var(x, axis)
    if keepdims:
        v = Nd4j.expandDims(s, axis)
    return v

@op
def std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    if axis is None:
        if keepdims:
            return Nd4j.std(x).reshape(* [1] * ndim(x))
        else:
            return Nd4j.std(x)
    s = Nd4j.var(x, axis)
    if keepdims:
        s = Nd4j.expandDims(s, axis)
    return s

@op
def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    """
    if axis is None:
        if keepdims:
            return Nd4j.mean(x).reshape(* [1] * ndim(x))
        else:
            return Nd4j.mean(x)
    m = Nd4j.mean(x, axis)
    if keepdims:
        m = Nd4j.expandDims(s, axis)
    return m

@op
def any(x, axis=None, keepdims=False):
    """Bitwise reduction (logical OR).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    raise NotImplementedError

@op
def all(x, axis=None, keepdims=False):
    """Bitwise reduction (logical AND).

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.
        keepdims: whether the drop or broadcast the reduction axes.

    # Returns
        A uint8 tensor (0s and 1s).
    """
    raise NotImplementedError

@op
def argmax(x, axis=-1):
    """Returns the index of the maximum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis += ndim(x)
    return Nd4j.argMax(x, axis)

@op
def argmin(x, axis=-1):
    """Returns the index of the minimum value along an axis.

    # Arguments
        x: Tensor or variable.
        axis: axis along which to perform the reduction.

    # Returns
        A tensor.
    """
    if axis < 0:
        axis += ndim(x)
    return Nd4j.argmin(x, axis)

@op
def square(x):
    """Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return x.mul(x)

@op
def abs(x):
    """Element-wise absolute value.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return Transforms.abs(x)

@op
def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return Transforms.sqrt(x)


@op
def exp(x):
    """Element-wise exponential.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return Transforms.exp(x)

@op
def log(x):
    """Element-wise log.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return Transforms.log(x)

@op
def logsumexp(x, axis=None, keepdims=False):
    """Computes log(sum(exp(elements across dimensions of a tensor))).

    This function is more numerically stable than log(sum(exp(x))).
    It avoids overflows caused by taking the exp of large inputs and
    underflows caused by taking the log of small inputs.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to reduce over.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`, the reduced dimension is
            retained with length 1.

    # Returns
        The reduced tensor.
    """
    raise NotImplementedError

@op
def round(x):
    """Element-wise rounding to the closest integer.

    In case of tie, the rounding mode used is "half to even".

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return Transforms.round(x)

@op
def sign(x):
    """Element-wise sign.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return Transforms.sign(x)


@op
def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: Tensor or variable.
        a: Python integer.

    # Returns
        A tensor.
    """
    return Transforms.pow(x, a)

@op
def clip(x, min_value, max_value):
    """Element-wise value clipping.

    # Arguments
        x: Tensor or variable.
        min_value: Python float or integer.
        max_value: Python float or integer.

    # Returns
        A tensor.
    """
    sd = SameDiff.create()  # clip not available in Nd4j
    if max_value is not None and max_value < min_value:
        max_value = min_value
    if max_value is None:
        max_value = np.inf
    x = sdvar(x)
    return sd.clipByValue(x, min_value, max_value).eval()

@sdop
def equal(x, y):
    """Element-wise equality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.eq(x, y)

@op
def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.neq(x, y)

@op
def greater(x, y):
    """Element-wise truth value of (x > y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.gt(x, y)

@op
def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.gte(x, y)

@op
def less(x, y):
    """Element-wise truth value of (x < y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.lt(x, y)

@op
def less_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return sd.lte(x, y)

@op
def maximum(x, y):
    """Element-wise maximum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.max(x, y)

@op
def minimum(x, y):
    """Element-wise minimum of two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.min(x, y)

@op
def sin(x):
    """Computes sin of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.sin(x)

@op
def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return sd.cos(x)

@op
def normalize_batch_in_training(x, gamma, beta,
                                reduction_axes, epsilon=1e-3):
    raise NotImplementedError


@op
def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    """
    return sd.batcNorm(x, mean, var, gamma, beta)

@op
def concatenate(tensors, axis=-1):
    if axis < 0:
        axis += ndim(tensors[0])
    return sd.concat(axis, *tensors)

@op
def reshape(x, shape):
    """Reshapes a tensor to the specified shape.

    # Arguments
        x: Tensor or variable.
        shape: Target shape tuple.

    # Returns
        A tensor.
    """
    return sd.reshape(x, *shape)

@op
def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.

    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.

    # Returns
        A tensor.
    """
    return sd.permute(x, *pattern)

@op
def resize_images(x, height_factor, width_factor, data_format):
    raise NotImplementedError

@op
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    raise NotImplementedError

@op
def repeat_elements(x, rep, axis):
    """Repeats the elements of a tensor along an axis, like `np.repeat`.

    If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
    will have shape `(s1, s2 * rep, s3)`.

    # Arguments
        x: Tensor or variable.
        rep: Python integer, number of times to repeat.
        axis: Axis along which to repeat.

    # Returns
        A tensor.
    """
    for _ in range(rep):
        x = sd.repeat(x, axis)
    return x

@op
def repeat(x, n):
    """Repeats a 2D tensor.

    if `x` has shape (samples, dim) and `n` is `2`,
    the output will have shape `(samples, 2, dim)`.

    # Arguments
        x: Tensor or variable.
        n: Python integer, number of times to repeat.

    # Returns
        A tensor.
    """
    assert ndim(x) == 2
    x = repeat_elements(x, n, 1)
    new_shape = list(int_shape(x))
    new_shape.insert(1, n)
    x = sd.reshape(x, *new_shape)

@op
def arange(start, stop=None, step=1, dtype='int32'):
    """Creates a 1D tensor containing a sequence of integers.

    The function arguments use the same convention as
    Theano's arange: if only one argument is provided,
    it is in fact the "stop" argument and "start" is 0.

    The default type of the returned tensor is `'int32'` to
    match TensorFlow's default.

    # Arguments
        start: Start value.
        stop: Stop value.
        step: Difference between two successive values.
        dtype: Integer dtype to use.

    # Returns
        An integer tensor.

    """
    raise NotImplementedError


@op
def tile(x, n):
    """Creates a tensor by tiling `x` by `n`.

    # Arguments
        x: A tensor or variable
        n: A list of integer. The length must be the same as the number of
            dimensions in `x`.

    # Returns
        A tiled tensor.
    """
    if isinstance(n, int):
        n = [n]
    return sd.tile(x, n)

@op
def flatten(x):
    """Flatten a tensor.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor, reshaped into 1-D
    """
    s = int_shape(x)
    vec_size = 1
    for i in s:
        vec_size *= i
    return sd.reshape(x, vec_size)

@op
def batch_flatten(x):
    s = int_shape(x)
    vec_size = 1
    for i in s[1:]:
        vec_size *= i
    return sd.reshape(x, -1, vec_size)

@op
def expand_dims(x, axis=-1):
    """Adds a 1-sized dimension at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Position where to add a new axis.

    # Returns
        A tensor with expanded dimensions.
    """
    return sd.expandDims(x, axis)

@op
def squeeze(x, axis):
    """Removes a 1-dimension from the tensor at index "axis".

    # Arguments
        x: A tensor or variable.
        axis: Axis to drop.

    # Returns
        A tensor with the same data as `x` but reduced dimensions.
    """
    return sd.squeeze(x, axis)

@op
def temporal_padding(x, padding=(1, 1)):
    raise NotImplementedError


@op
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    raise NotImplementedError

@op
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    raise NotImplementedError

@op
def stack(x, axis=0):
    """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

    # Arguments
        x: List of tensors.
        axis: Axis along which to perform stacking.

    # Returns
        A tensor.
    """
    return tf.stack(x, axis)


@op
def one_hot(indices, num_classes):
    """Computes the one-hot representation of an integer tensor.

    # Arguments
        indices: nD integer tensor of shape
            `(batch_size, dim1, dim2, ... dim(n-1))`
        num_classes: Integer, number of classes to consider.

    # Returns
        (n + 1)D one hot representation of the input
        with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`
    """
    return sd.oneHot(indices, depth=num_classes)


@op
def bias_add(x, bias, data_format=None):
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))
    bias_ndim = ndim(bias)
    x_ndim = ndim(x)
    if bias_ndim != 1 and bias_ndim != x_ndim - 1:
        raise ValueError('Unexpected bias dimensions %d, '
                         'expect to be 1 or %d dimensions'
                         % (bias_ndim, x_ndim - 1))
    bias_shape = tuple(bias.shape())
    if x_ndim == 5:
        if data_format == 'channels_first':
            if bias_ndim == 1:
                x.addi(bias.reshape(1, bias_shape[0], 1, 1, 1))
                #x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
            else:
                x.addi(bias.rehsape(1, bias_shape[3], *bias_shape[:3]))
                #x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
        elif data_format == 'channels_last':
            if bias_ndim == 1:
                x.addi(bias.reshape(1, 1, 1, 1, bias_shape[0]))
                #x += reshape(bias, (1, 1, 1, 1, bias_shape[0]))
            else:
                x.addi(bias.rehsape(1, *bias_shape))
               # x += reshape(bias, (1,) + bias_shape)
    elif x_ndim == 4:
        if data_format == 'channels_first':
            if bias_ndim == 1:
                x.addi(bias.reshape(bias, 1, bias_shape[0], 1, 1))
                #x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x.addi(bias.reshape(1, bias_shape[2], *bias_shape[:2]))
                #x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if ndim(bias) == 1:
                x.addi(bias.rehsape(1, 1, 1, bias_shape[0]))
                #x += reshape(bias, (1, 1, 1, bias_shape[0]))
            else:
                x.addi(bias.reshape(1, *bias_shape))
                #x += reshape(bias, (1,) + bias_shape)
    elif x_ndim == 3:
        if data_format == 'channels_first':
            if bias_ndim == 1:
                x.addi(bias.rehsape(1, bias_shape[0], 1))
                #x += reshape(bias, (1, bias_shape[0], 1))
            else:
                x.addi(bias.reshape(1, bias_shape[1], bias_shape[0]))
                #x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
        elif data_format == 'channels_last':
            if bias_ndim == 1:
                x.addi(bias.reshape(1, 1, bias_shape[0]))
                #x += reshape(bias, (1, 1, bias_shape[0]))
            else:
                x.addi(bias.reshape(1, *bias_shape))
                #x += reshape(bias, (1,) + bias_shape)
    else:
        x.addi(bias)
    return x



###-----------NO OPS BEYOND THIS LINE---------------###


def function(inputs, outputs, updates=None, **kwargs):
    return Graph(inputs, outputs)
