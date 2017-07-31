"""
Setup the Inception v3 models from Keras and load weights.

Create a base model with my input shape. Then add the layers on top
necessary for multi class classification with 17 labels.
Code based on the example in the Keras applications.
"""

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D


def get_models():
    """Return the base model and our modified model to fit."""
    # Create the base pre-trained model
    base_model = InceptionV3(input_shape=(256, 256, 3),
                             weights='imagenet',
                             include_top=False)

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # And a classifier layer -- we have 17 classes
    # The sigmoid allows for multiclass labelling.
    predictions = Dense(17, activation='sigmoid')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return base_model, model
