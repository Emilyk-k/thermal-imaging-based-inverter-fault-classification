classdef config
    properties (Constant)
        TRAIN_DATA = fullfile(pwd, 'data', 'article_explanation');
        TEST_DATA = fullfile(pwd, 'data', 'article_explanation');
        ONNX_PATH = fullfile(pwd, 'models', 'EfficientNetV2L_imagenet_640x480_noTop_poolingAvg.onnx');
        MODEL_SAVE_PATH = fullfile(pwd, 'models', 'fault_detection_model_single_speed_40_classes_transient.mat');
        HISTOGRAM_SAVE_PATH = fullfile(pwd, 'results');
        EXTENSION = {'.png'};
        
        IMG_HEIGHT = 640;
        IMG_WIDTH = 480;
        CHANNELS = 3;
        NUM_CLASSES = 40; % 7 -> single key failure
        NORMALIZATION = false;
        
        BATCH_SIZE = 8;
        EPOCHS = 12;
        LEARNING_RATE = 1e-5;
        LR_DROP_PERIOD = 0;
        NUM_FROZEN = 50; % training.m adds 2 when using normalization
        VALIDATION_SPLIT = 0.2;
        PATIENCE = 30;
        GRADIENT_DEC_FAC = 0.9;
        SQUARED_GRADIENT_DEC_FAC = 0.999;
        EPSILON = 1e-8;
        VALIDATION_FREQ = 40;
    end
end