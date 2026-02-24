import numpy as np
import tensorflow as tf
from paz.models.detection.dino_v2 import DINOv2ViTS


def get_optimizer(learning_rate=1e-4):
    """Creates an Adam optimizer for training DINOv2.

    # Arguments
        learning_rate: Float, the learning rate.

    # Returns
        optimizer: Adam optimizer instance.
    """
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def train(num_classes=80, input_shape=(560, 560, 3), epochs=10,
          batch_size=4, learning_rate=1e-4):
    """Training script for DINOv2 ViT-S object detection.

    # Arguments
        num_classes: Int, number of object classes.
        input_shape: Tuple, input image shape (H, W, C).
        epochs: Int, number of training epochs.
        batch_size: Int, number of samples per batch.
        learning_rate: Float, learning rate for Adam optimizer.
    """
    model = DINOv2ViTS(num_classes=num_classes, input_shape=input_shape)
    optimizer = get_optimizer(learning_rate)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()
    print('Model created successfully with output shape:', model.output_shape)
    return model


if __name__ == '__main__':
    model = train()
