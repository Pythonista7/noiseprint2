import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from trainViT import setup_session, create_vit_classifier

list_classes = ['iPhone-4s',
                'Sony-NEX-7',
                'iPhone-6',
                'Samsung-Galaxy-Note3',
                'Motorola-Droid-Maxx',
                'Motorola-Nexus-6',
                'Samsung-Galaxy-S4',
                'LG-Nexus-5x',
                'Motorola-X',
                'HTC-1-M7']


def read_and_resize(filepath):
    im_array = np.array(Image.open((filepath)), dtype="uint8")
    pil_im = Image.fromarray(im_array)
    new_array = np.array(pil_im.resize((256, 256)))
    return new_array/255


def label_transform(labels):
    labels = pd.get_dummies(pd.Series(labels))
    label_index = labels.columns.values
    return labels, label_index


def preprocess_input(img_path):
    img = read_and_resize(img_path)
    return img


if __name__ == '__main__':
    checkpoint_filepath = "/tmp/checkpoint"
    model = create_vit_classifier()
    model.load_weights(checkpoint_filepath)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=0.001, weight_decay=0.0001
    )
    print("Optimizer is ready .. ")

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(
                5, name="top-5-accuracy"),
        ],
    )
    print("weights loaded")
    sample = preprocess_input(
        "/home/ash/Desktop/7thSem/7thsem/FinalYearProj/exp/data/iPhone-4s/14-b20.jpg")
    logits = model.evaluate(x=sample, batch_size=1)
    print(logits)
