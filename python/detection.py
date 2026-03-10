import tensorflow as tf
import os
from config import *


def load_trained_model():
    model_path = MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        final_model_path = MODEL_SAVE_PATH.replace('.h5', '_final.h5')
        if os.path.exists(final_model_path):
            model_path = final_model_path
        else:
            raise FileNotFoundError(f"Missing model: {model_path}")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded")

    return model


def evaluate_model(model):
    from preprocessing import get_test_generator

    try:
        test_generator = get_test_generator()
        results = model.evaluate(test_generator, verbose=1)

        print("\n" + "=" * 30)
        print("Results on test set")
        print("=" * 30)
        print(f"Loss: {results[0]:.4f}")
        print(f"Accuracy: {results[1]:.4f}")
        if len(results) > 2:
            print(f"Precision: {results[2]:.4f}")
            print(f"Recall: {results[3]:.4f}")

        return results

    except Exception as e:
        print(f"Evaluation error: {e}")
        return None
