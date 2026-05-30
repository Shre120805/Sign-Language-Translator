import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2

INPUT_DIM = 63   # 21 landmarks × 3 (x, y, z)


def build_classifier(num_classes, input_dim=INPUT_DIM):
    """
    Build MLP classifier for hand landmarks.

    Args:
        num_classes: Number of sign classes
        input_dim: Feature dimensionality (default 63 for 21 landmarks × 3)

    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(input_dim,))

    # Hidden layer 1
    x = layers.Dense(256, kernel_regularizer=l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Hidden layer 2
    x = layers.Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    # Hidden layer 3
    x = layers.Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="LandmarkClassifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    model = build_classifier(num_classes=28)
    model.summary()
