import datetime
import logging

from PIL import Image
from tensorflow.python.keras.backend import resize_images
from tensorflow.python.keras.engine.input_layer import Input, InputLayer
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Activation

import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers.convolutional import Cropping2D
from tensorflow.python.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.image_ops_impl import crop_and_resize_v1

from tensorflow.keras import models
from tensorflow import keras
import tensorflow_addons as tfa


from noiseprint2.utility import jpeg_quality_of_file

from trainViT import create_vit_classifier

# Bias layer necessary because noiseprint applies bias after batch-normalization.


class BiasLayer(tf.keras.layers.Layer):
    """
    Simple bias layer
    """

    def build(self, input_shape):
        self.bias = self.add_weight(
            'bias', shape=input_shape[-1], initializer="zeros")

    @tf.function
    def call(self, inputs, training=None):
        return inputs + self.bias


def _full_conv_net(num_levels=17, padding='SAME'):
    """FullConvNet model."""
    activation_fun = [tf.nn.relu, ] * (num_levels - 1) + [tf.identity, ]
    filters_num = [64, ] * (num_levels - 1) + [1, ]
    batch_norm = [False, ] + [True, ] * (num_levels - 2) + [False, ]

    inp = tf.keras.layers.Input([None, None, 1])
    model = inp

    for i in range(num_levels):
        model = Conv2D(filters_num[i], 3,
                       padding=padding, use_bias=False)(model)
        if batch_norm[i]:
            model = BatchNormalization(epsilon=1e-5)(model)
        model = BiasLayer()(model)
        model = Activation(activation_fun[i])(model)

    return Model(inp, model)


def setup_session():
    """
    Set the session allow_growth option for GPU usage
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)


def addEnsenble(noiseprint_model):
    #Add noiseprint model and freeze training for this stack
    noiseprint_model.trainable = False
    model = models.Sequential()
    model.add(noiseprint_model)
    #print("Adding input crop")
    #model.add(Cropping2D(cropping=256))
    print("Adding ViT ... ")
    vit = create_vit_classifier()
    model.add(vit)
    return model


class NoiseprintEngine:
    _save_path = os.path.join(os.path.dirname(
        __file__), './weights/net_jpg%d/')
    slide = 1024  # 3072
    large_limit = 1050000  # 9437184
    overlap = 34
    setup_on_init = True

    def __init__(self, train=False):
        self._model = _full_conv_net()
        self._loaded_quality = 90
        checkpoint = self._save_path % self._loaded_quality
        self._model.load_weights(checkpoint)
        self._model = addEnsenble(self._model)
        if self.setup_on_init:
            setup_session()
            if train == True:
                self.train_ensemble()

    def train_ensemble(self):
        X = np.load("../dat/X_train_grayscale.npy")
        y = np.load("../dat/label_y.npy")

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

        print(
            f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
        print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

        print("Creating checkpoints .. ")
        checkpoint_filepath = "/tmp/checkpoint-ensemble"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
        learning_rate = 0.001
        weight_decay = 0.0001

        optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
        )
        print("Optimizer is ready .. ")

        self._model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )
        print("Model is complied ")

        log_dir = "logs/fit/ensemble1-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

        print("Running model.fit ...")
        print("xtrain  shape = ", x_train.shape)


        history = self._model.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
            epochs=1,
            validation_split=0.33,
            callbacks=[checkpoint_callback, tensorboard_callback]
        )

        self._model.load_weights(checkpoint_filepath)

        print(f"Running test")
        _, accuracy, top_5_accuracy = self._model.evaluate(x_test, y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
        return history

    def load_quality(self, quality):
        """
        Loads a quality level for the next noiseprint predictions.
        Quality level can be obtained by the file quantization table.
        :param quality: Quality level, int between 51 and 101 (included)
        """
        if quality < 51 or quality > 101:
            raise ValueError(
                "Quality must be between 51 and 101 (included). Provided quality: %d" % quality)
        if quality == self._loaded_quality:
            return
        logging.info("Loading checkpoint quality %d" % quality)
        checkpoint = self._save_path % quality
        self._model.load_weights(checkpoint)
        self._loaded_quality = quality

    @tf.function(experimental_relax_shapes=True,
                 input_signature=[tf.TensorSpec(shape=(1, None, None, 1), dtype=tf.float32)])
    def _predict(self, img):
        return self._model(img)

    def _predict_small(self, img):
        return np.squeeze(self._predict(img[np.newaxis, :, :, np.newaxis]).numpy())

    def _predict_large(self, img):
        # prepare output array
        res = np.zeros((img.shape[0], img.shape[1]), np.float32)

        # iterate over x and y, strides = self.slide, window size = self.slide+2*self.overlap
        for x in range(0, img.shape[0], self.slide):
            x_start = x - self.overlap
            x_end = x + self.slide + self.overlap
            for y in range(0, img.shape[1], self.slide):
                y_start = y - self.overlap
                y_end = y + self.slide + self.overlap
                patch = img[max(x_start, 0): min(x_end, img.shape[0]), max(
                    y_start, 0): min(y_end, img.shape[1])]
                patch_res = np.squeeze(self._predict_small(patch))

                # discard initial overlap if not the row or first column
                if x > 0:
                    patch_res = patch_res[self.overlap:, :]
                if y > 0:
                    patch_res = patch_res[:, self.overlap:]
                # discard data beyond image size
                patch_res = patch_res[:min(self.slide, patch_res.shape[0]), :min(
                    self.slide, patch_res.shape[1])]
                # copy data to output buffer
                res[x: min(x + self.slide, res.shape[0]),
                    y: min(y + self.slide, res.shape[1])] = patch_res
        return res

    def predict(self, img):
        """
        Run the noiseprint generation CNN over the input image
        :param img: input image, 2-D numpy array
        :return: output noisepritn, 2-D numpy array with the same size of the input image
        """
        if len(img.shape) != 2:
            raise ValueError(
                "Input image must be 2-dimensional. Passed shape: %r" % img.shape)
        if self._loaded_quality is None:
            raise RuntimeError(
                "The engine quality has not been specified, please call load_quality first")
        if img.shape[0] * img.shape[1] > self.large_limit:
            return self._predict_large(img)
        else:
            return self._predict_small(img)


def gen_noiseprint(image, quality=None):
    """
    Generates the noiseprint of an image
    :param image: image data. Numpy 2-D array or path string of the image
    :param quality: Desired quality level for the noiseprint computation.
    If not specified the level is extracted from the file if image is a path string to a JPEG file, else 101.
    :return: The noiseprint of the input image
    """
    if isinstance(image, str):
        # if image is a path string, load the image and quality if not defined
        if quality is None:
            try:
                quality = jpeg_quality_of_file(image)
            except AttributeError:
                quality = 101
        image = np.asarray(Image.open(image).convert(
            "YCbCr"))[..., 0].astype(np.float32) / 256.0
    else:
        if quality is None:
            quality = 101
    engine = NoiseprintEngine()
    engine.load_quality(quality)
    return engine.predict(image)


def normalize_noiseprint(noiseprint, margin=34):
    """
    Normalize the noiseprint between 0 and 1, in respect to the central area
    :param noiseprint: noiseprint data, 2-D numpy array
    :param margin: margin size defining the central area, default to the overlap size 34
    :return: the normalized noiseprint data, 2-D numpy array with the same size of the input noiseprint data
    """
    v_min = np.min(noiseprint[margin:-margin, margin:-margin])
    v_max = np.max(noiseprint[margin:-margin, margin:-margin])
    return ((noiseprint - v_min) / (v_max - v_min)).clip(0, 1)
