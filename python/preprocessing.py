import tensorflow as tf
from config import *


def create_data_generators():
    # Gatagen for training/validation
    print("Generating data...")

    # Datagen with optional augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=VALIDATION_SPLIT,
        rescale=1./255
    )

    # Training data
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        color_mode = 'rgb'
    )

    # Validation data
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DATA,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        color_mode='rgb'
    )

    print(TRAIN_DATA)

    print(f"Classes found: {train_generator.class_indices}")
    print(f"Training samples amount: {train_generator.samples}")
    print(f"Validation samples amount: {validation_generator.samples}")

    return train_generator, validation_generator


def get_test_generator():
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        TEST_DATA,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )

    return test_generator