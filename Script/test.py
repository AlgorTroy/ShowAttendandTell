import tensorflow as tf

"""VGG16 model.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""

img_tensor = None


def vgg16(input_tensor=None, input_shape=None, pooling=None, classes=None):
    """ VGG16 Net Arch implementing Tensorflow
    # Arguments:
        input_tensor : image tensor
        input_shape : shape of the tensor
        pooling: optional pooling
        classes: optional classes number

    # Returns:
        Tensorflow VGG16 instance
    """
