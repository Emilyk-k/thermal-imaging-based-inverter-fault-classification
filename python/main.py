import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from training import *
from detection import *

def main():
    print("START TRAIN")
    train_model(ADVANCED)
    print("FINISH TRAIN")

    print("START TEST")
    model = load_trained_model()
    evaluate_model(model) # Evaluation on TEST_DATA
    print("FINISH TEST")


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to the first GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

        main()
    except RuntimeError as e:
        print(e)
else:
    print("No gpu found")
    main()