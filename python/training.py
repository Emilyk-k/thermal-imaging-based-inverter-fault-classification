import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from config import *
from preprocessing import create_data_generators


def build_advanced_model():
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights="imagenet", include_top=False,
                                                                       input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
                                                                       pooling="avg")
    base_model.trainable = True
    frozen_num_layer = NUM_FROZEN
    print(f'{frozen_num_layer} layes frozen out of  {str(len(base_model.layers))} in base model')

    for layer in base_model.layers[:frozen_num_layer]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")
    ])

    return model


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    return model


def setup_callbacks():
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=LR_MIN,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    return callbacks


def plot_training_history(history, epochs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Dokładność
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Strata
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Precyzja
    if 'precision' in history.history:
        ax3.plot(history.history['precision'], label='Training', linewidth=2)
        ax3.plot(history.history['val_precision'], label='Validation', linewidth=2)
        ax3.set_title('Model precision', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Precision')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Czułość
    if 'recall' in history.history:
        ax4.plot(history.history['recall'], label='Training', linewidth=2)
        ax4.plot(history.history['val_recall'], label='Validation', linewidth=2)
        ax4.set_title('Model recall', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    print(history)
    print(range(epochs))

    os.makedirs(HISTOGRAM_SAVE_PATH, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(HISTOGRAM_SAVE_PATH) + "training_hist.png")
    print('histogram saved')
    plt.show()


def train_model(use_advanced_model=True):
    print("--- TRAINING STARTED ---")

    train_generator, validation_generator = create_data_generators()
    print("img_input_size HxW: " + str(IMG_HEIGHT) + 'x' + str(IMG_WIDTH))
    print("batch_size: " + str(BATCH_SIZE))

    if use_advanced_model:
        print("Building advanced CNN model...")
        model = build_advanced_model()
    else:
        print("Building simple CNN model...")
        model = build_simple_model()

    model = compile_model(model)
    print('number of layers in model ' + str(len(model.layers)))
    print(model.summary())

    callbacks = setup_callbacks()

    validation_steps = validation_generator.n // BATCH_SIZE
    steps_per_epoch = train_generator.n // BATCH_SIZE
    epochs = 200

    print('Epochs, steps per epoch, validation steps: ', epochs, steps_per_epoch, validation_steps, sep=' ')

    print("started training model")
    start = time.time()

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    end = time.time()
    real_time = end-start

    print("finished training model")
    print("real_time in second: ", str(real_time))

    model.save(MODEL_SAVE_PATH.replace('.h5', '_final.h5'))
    print('model saved')

    plot_training_history(history, EPOCHS)

    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

    return history, model
