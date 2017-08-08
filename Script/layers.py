import tensorflow as tf


class layers():
    def conv_layer(input_tensor, conv_stride, conv_padding, kernel_shape, kernel_stddev, name_scope=None, dtype=tf.float32):
        """

        # Arguments:
        :param input_tensor: tensor to be processed
        :param conv_stride: stride with which the filter need to move
        :param conv_padding: padding
        :param kernel_shape: a list
        :param kernel_stddev: standard_deviation
        :param name_scope: scope
        :param dtype: default tf.float32
        :return: output, filter, biases
        """
        with tf.name_scope(name_scope) as scope:
            kernel = tf.Variable(tf.truncated_normal(kernel_shape, dtype=dtype,
                                                     stddev=kernel_stddev), name='weights')
            conv = tf.nn.conv2d(input_tensor, kernel, conv_stride, padding=conv_padding)
            biases = tf.Variable(tf.constant(0.0, shape=[kernel_shape[-1]], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
        return out, kernel, biases