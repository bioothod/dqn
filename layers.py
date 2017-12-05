import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import standard_ops

import math

class Noise(base.Layer):
    def __init__(self, units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               **kwargs):
        super(Noise, self).__init__(trainable=trainable, name=name,
                                #activity_regularizer=activity_regularizer,
                                **kwargs)
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_spec = base.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Noise` '
                    'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})

        boundary = tf.sqrt(1. / float(input_shape[-1].value))
        kninit = tf.random_uniform_initializer(-boundary, boundary)

        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=kninit,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)


        self.kernel_sigma = self.add_variable('kernel_sigma',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=tf.constant_initializer(0.4/math.sqrt(float(input_shape[-1].value))),
                                        dtype=self.dtype,
                                        trainable=True)

        self.kernel_eps_in = self.add_variable('kernel_eps_in',
                                        shape=[input_shape[-1].value, 1],
                                        initializer=tf.zeros_initializer,
                                        dtype=self.dtype,
                                        trainable=False)
        self.kernel_eps_out = self.add_variable('kernel_eps_out',
                                        shape=[1, self.units],
                                        initializer=tf.zeros_initializer,
                                        dtype=self.dtype,
                                        trainable=False)
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[self.units,],
                                          initializer=kninit,
                                          regularizer=self.bias_regularizer,
                                          constraint=self.bias_constraint,
                                          dtype=self.dtype,
                                          trainable=True)

            self.bias_sigma = self.add_variable('bias_sigma',
                                                shape=[self.units,],
                                                initializer=tf.constant_initializer(0.4/math.sqrt(float(input_shape[-1].value))),
                                                dtype=self.dtype,
                                                trainable=True)

        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        func = lambda x: tf.sign(x) * tf.sqrt(tf.abs(x))
        self.kernel_eps_in = func(tf.random_normal(self.kernel_eps_in.shape))
        self.kernel_eps_out = func(tf.random_normal(self.kernel_eps_out.shape))
        if self.use_bias:
            self.bias_eps = tf.random_normal(self.bias_eps.shape)

        eps = tf.multiply(self.kernel_eps_out, self.kernel_eps_in)

        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel + self.kernel_sigma * eps,
                    [[len(shape) - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if context.in_graph_mode():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = standard_ops.matmul(inputs, self.kernel + self.kernel_sigma * eps)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias + self.bias_sigma * self.kernel_eps_out)

        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable

        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if input_shape[-1].value is None:
            raise ValueError('The innermost dimension of input_shape must be defined, but saw: %s' % input_shape)

        return input_shape[:-1].concatenate(self.units)

def noise(inputs, units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=init_ops.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None):
    layer = Noise(units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                trainable=trainable,
                name=name,
                dtype=inputs.dtype.base_dtype,
                _scope=name,
                _reuse=reuse)
    return layer.apply(inputs)
