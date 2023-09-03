from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, GlobalAveragePooling2D,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax

from config import Config


def create_model(model_class=ResNet50):
    """
    Create a custom model based on a model class.

    Args:
        model_class (tf.keras.Model): Model class.

    Returns:
        model (tf.keras.Model): Custom model.
    """
    # Load the pretrained model
    img_shape = (Config.input_size[0], Config.input_size[0], 3)
    base_model = ResNet50(weights=Config.weights, include_top=False, input_shape=img_shape)
    base_model.trainable = False

    # Define the additional convolutional block
    conv_block = Sequential([
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        GlobalAveragePooling2D()
    ])

    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        conv_block,  # Add the custom convolutional block here
        Dense(256, kernel_regularizer=regularizers.l2(l=0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(2, activation='softmax')
    ])

    # Compile the model
    model.compile(Adamax(Config.learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
