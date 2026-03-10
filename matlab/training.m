function [history, trainedNet] = training()
    
    try
        gpuAvailable = gpuDeviceCount > 0;
        if gpuAvailable
            gpu = gpuDevice(1);
            fprintf('GPU detected: %s\n', gpu.Name);
            fprintf('GPU Memory: %.0f MB\n', gpu.AvailableMemory/1000000);
            
            reset(gpu);
            executionEnv = 'gpu';
        else
            fprintf('----------No GPU available, using CPU----------\n');
            executionEnv = 'cpu';
        end
    catch
        fprintf('GPU check failed, defaulting to CPU\n');
        executionEnv = 'cpu';
        gpuAvailable = false;
    end
    
    fprintf('Generating data\n');

    imds = imageDatastore(config.TRAIN_DATA, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', config.EXTENSION);

    [trainImds, valImds] = splitEachLabel(imds, 1 - config.VALIDATION_SPLIT, 'randomized');

    trainDatastore = augmentedImageDatastore([config.IMG_HEIGHT, config.IMG_WIDTH], ...
        trainImds, ...
        'ColorPreprocessing', 'gray2rgb');

    valDatastore = augmentedImageDatastore([config.IMG_HEIGHT, config.IMG_WIDTH], ...
        valImds, ...
        'ColorPreprocessing', 'gray2rgb');

    classNames = categories(trainImds.Labels);

    fprintf('Training directory: %s\n', config.TRAIN_DATA);
    fprintf('Classes found: %s\n', strjoin(classNames, ', '));
    fprintf('Training samples: %d\n', numel(trainImds.Files));
    fprintf('Validation samples: %d\n', numel(valImds.Files));
    fprintf('Image input size: %dx%d\n', config.IMG_HEIGHT, config.IMG_WIDTH);
    fprintf('Batch size: %d\n', config.BATCH_SIZE);

    fprintf('Building advanced model...\n');
    model = build_model(trainImds);

    analyzeNetwork(model);

    numTrainingObservations = numel(trainImds.Files);

    iterationsPerEpoch = ceil(numTrainingObservations / config.BATCH_SIZE);
    
    if config.LR_DROP_PERIOD ~= 0
        options = trainingOptions('adam', ...
        'InitialLearnRate', config.LEARNING_RATE, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.2, ... 
        'LearnRateDropPeriod', config.LR_DROP_PERIOD, ...  
        'MaxEpochs', config.EPOCHS, ...
        'MiniBatchSize', config.BATCH_SIZE, ...
        'ValidationData', valDatastore, ...
        'ValidationFrequency', config.VALIDATION_FREQ, ... 
        'ValidationPatience', config.PATIENCE, ...
        'GradientDecayFactor', config.GRADIENT_DEC_FAC, ...
        'SquaredGradientDecayFactor', config.SQUARED_GRADIENT_DEC_FAC, ...
        'Epsilon', config.EPSILON, ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', executionEnv, ...
        'Shuffle', 'every-epoch', ...
        'DispatchInBackground', gpuAvailable);
    else
        options = trainingOptions('adam', ...
        'InitialLearnRate', config.LEARNING_RATE, ... 
        'MaxEpochs', config.EPOCHS, ...
        'MiniBatchSize', config.BATCH_SIZE, ...
        'ValidationData', valDatastore, ...
        'ValidationFrequency', config.VALIDATION_FREQ, ... 
        'ValidationPatience', config.PATIENCE, ...
        'GradientDecayFactor', config.GRADIENT_DEC_FAC, ...
        'SquaredGradientDecayFactor', config.SQUARED_GRADIENT_DEC_FAC, ...
        'Epsilon', config.EPSILON, ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', executionEnv, ...
        'Shuffle', 'every-epoch', ...
        'DispatchInBackground', gpuAvailable);
    end

    fprintf('Training started\n');
    fprintf('Using %s for training\n', upper(executionEnv));
    
    startTime = tic;

    [trainedNet, info] = trainNetwork(trainDatastore, model, options);

    resultsDir = './results';
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end
    
    save(fullfile(resultsDir, 'training_info.mat'), 'info');

    trainingTime = toc(startTime);
    fprintf('Finished training\n');
    fprintf('Training time: %.2f h, %.2f min, %.2f s\n', trainingTime/3660, trainingTime/60, trainingTime);

    save(config.MODEL_SAVE_PATH, 'trainedNet'); % MATLAB saves only the final model
    fprintf('Model saved\n');

    plot_training_history(info);

    fprintf('Final validation accuracy: %.4f\n', info.ValidationAccuracy(end));
    fprintf('Best validation accuracy: %.4f\n', max(info.ValidationAccuracy(~isnan(info.ValidationAccuracy))));

    history = info;
end

function model = build_model(trainImds)
    
    fprintf('Building EfficientNet-B0 with 640x480 input\n');
    
    inputSize = [config.IMG_HEIGHT, config.IMG_WIDTH, config.CHANNELS];
    numClasses = config.NUM_CLASSES;
    
    gpuAvailable = gpuDeviceCount > 0;
    if gpuAvailable
        fprintf('GPU detected, optimizing model for GPU training...\n');
    end
    
    net = imagePretrainedNetwork('efficientnetb0', 'NumClasses', numClasses, 'Weights', 'pretrained');
    
    lgraph = layerGraph(net);
    
    fprintf('Replacing input layer...\n');
    
    if ismember('ImageInput', {lgraph.Layers.Name})
        lgraph = removeLayers(lgraph, 'ImageInput');
        fprintf('Removed: ImageInput\n');
    end

    if config.NORMALIZATION == true
        fprintf('Calculating dataset statistics...\n');
        allImages = readall(trainImds); 
        allImagesData = double(cat(4, allImages{:})) / 255; 
        
        m = mean(allImagesData, 'all');
        s = std(allImagesData, 0, 'all');
        
        datasetMean = reshape([m, m, m], [1 1 3]);
        datasetStd  = reshape([s, s, s], [1 1 3]);
        
        fprintf('Dataset Mean (expanded to 3ch): %s\n', num2str(squeeze(datasetMean)', '%.4f  '));
        fprintf('Dataset Std (expanded to 3ch): %s\n', num2str(squeeze(datasetStd)', '%.4f  '));
    
        inputLayer = imageInputLayer(inputSize, 'Name', 'ImageInputRaw', 'Normalization', 'none');
        rescaleLayer = scalingLayer('Name', 'Rescale01', 'Scale', 1/255);
        
        zscoreLayer = batchNormalizationLayer('Name', 'DatasetZScore', ...
            'Offset', zeros(1, 1, 3), ...
            'Scale', ones(1, 1, 3), ...
            'TrainedMean', datasetMean, ...
            'TrainedVariance', datasetStd.^2, ...
            'Epsilon', 1e-5);

        fprintf('Added normalization: ImageInputLayer, RescaleLayer, zScoreLayer\n')
    
        lgraph = addLayers(lgraph, [inputLayer; rescaleLayer; zscoreLayer]);
    
        firstConvLayer = 'efficientnet-b0|model|stem|conv2d|Conv2D';
        lgraph = connectLayers(lgraph, 'DatasetZScore', firstConvLayer);

    else
        newInput = imageInputLayer(inputSize, 'Name', 'ImageInputCustom', 'Normalization', 'none');
        lgraph = addLayers(lgraph, newInput);
        
        firstConvLayer = 'efficientnet-b0|model|stem|conv2d|Conv2D';
        if ismember(firstConvLayer, {lgraph.Layers.Name})
            lgraph = connectLayers(lgraph, 'ImageInputCustom', firstConvLayer);
            fprintf('Connected to: %s\n', firstConvLayer);
            fprintf('Normalization omitted \n')
        else
            error('Could not find first conv layer: %s', firstConvLayer);
        end
    end
    
    fprintf('Removing original head and output layers\n');
    
    layerNames = {lgraph.Layers.Name};
    
    layersToRemove = layerNames( ...
        contains(layerNames, '|head|') | ...
        strcmp(layerNames, 'Softmax') | ...
        strcmp(layerNames, 'ClassificationLayer') ...
    );
    
    if isempty(layersToRemove)
        error('No head/output layers found to remove.');
    end
    
    for i = 1:numel(layersToRemove)
        lgraph = removeLayers(lgraph, layersToRemove{i});
        fprintf('   Removed: %s\n', layersToRemove{i});
    end

    fprintf('Finding last feature layer by graph connectivity...\n');
    
    connections = lgraph.Connections;
    allSources = connections.Source;
    
    layerNames = {lgraph.Layers.Name};
    
    candidateLayers = setdiff(layerNames, allSources);
    
    if numel(candidateLayers) ~= 1
        error('Expected exactly one terminal layer, found %d', numel(candidateLayers));
    end
    
    lastLayer = candidateLayers{1};
    fprintf('   Using terminal feature layer: %s\n', lastLayer);


    fprintf('Adding custom classification head...\n');

    customHead = [
        fullyConnectedLayer(numClasses, 'Name', 'prediction_dense')
        softmaxLayer('Name', 'prediction_softmax')
        classificationLayer('Name', 'output')
    ];

    lgraph = addLayers(lgraph, customHead);
    lgraph = connectLayers(lgraph, lastLayer, 'prediction_dense');

    fprintf('   Added custom head: Dense(%d) → Softmax → Classification\n', numClasses);

    fprintf('Freezing layers for transfer learning...\n');
    
    if config.NORMALIZATION == true
        numToFreeze = min(config.NUM_FROZEN+2, numel(lgraph.Layers));
    else
        numToFreeze = min(config.NUM_FROZEN, numel(lgraph.Layers));
    end

    frozenIndices = [];

    for i = 1:numToFreeze
        layer = lgraph.Layers(i);
        modified = false;
        
        if isprop(layer, 'WeightLearnRateFactor')
            layer.WeightLearnRateFactor = 0;
            layer.BiasLearnRateFactor = 0;
            modified = true;
        end
        
        if isprop(layer, 'ScaleLearnRateFactor')
            layer.ScaleLearnRateFactor = 0;
            layer.OffsetLearnRateFactor = 0;
            modified = true;
        end
        
        if modified
            lgraph = replaceLayer(lgraph, layer.Name, layer);
            frozenIndices = [frozenIndices, i];
        end
    end
    
    numFrozenLayers = numel(frozenIndices);
    numTotalLayers = numel(lgraph.Layers);
    
    fprintf('   Checked first %d layers.\n', numToFreeze);
    fprintf('   Modified %d learnable layers.\n', numFrozenLayers);
    fprintf('   Indices of modified layers: %s\n', mat2str(frozenIndices));

    frozenLayers = [];
    for i = 1:numel(lgraph.Layers)
        layer = lgraph.Layers(i);
        if isprop(layer, 'WeightLearnRateFactor')
            if layer.WeightLearnRateFactor == 0
                frozenLayers = [frozenLayers, i];
            end
        end
    end
    
    fprintf('   Verified frozen layers: %s\n', mat2str(frozenLayers));
    
    model = lgraph;
    
    fprintf('\n=== MODEL ARCHITECTURE ===\n');
    fprintf('Input:  %dx%dx%d\n', inputSize(1), inputSize(2), inputSize(3));
    fprintf('Base:   EfficientNet-B0\n');
    fprintf('Layers: %d Total\n', numTotalLayers);
    fprintf('        %d Trainable layers frozen (indices 1 to %d)\n', numFrozenLayers, numToFreeze);
    fprintf('===========================\n');
end


function plot_training_history(info)

    fig = figure('Position', [100, 100, 1000, 400]);
    
    subplot(1, 2, 1);
    plot(info.TrainingAccuracy, 'b-', 'LineWidth', 1);
    hold on;
    valIdxAcc = find(~isnan(info.ValidationAccuracy));
    plot(valIdxAcc, info.ValidationAccuracy(valIdxAcc), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    
    title('Model Accuracy', 'FontSize', 12);
    xlabel('Iteration');
    ylabel('Accuracy (%)');
    legend('Training', 'Validation', 'Location', 'SouthEast');
    grid on;
    hold off;

    subplot(1, 2, 2);
    plot(info.TrainingLoss, 'b-', 'LineWidth', 1);
    hold on;
    valIdxLoss = find(~isnan(info.ValidationLoss));
    plot(valIdxLoss, info.ValidationLoss(valIdxLoss), 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    
    title('Model Loss', 'FontSize', 12);
    xlabel('Iteration');
    ylabel('Loss');
    legend('Training', 'Validation', 'Location', 'NorthEast');
    grid on;
    hold off;

    if ~exist(config.HISTOGRAM_SAVE_PATH, 'dir')
        mkdir(config.HISTOGRAM_SAVE_PATH);
    end
    saveas(fig, fullfile(config.HISTOGRAM_SAVE_PATH, 'training_history.png'));
    fprintf('Training history plot saved to: %s\n', config.HISTOGRAM_SAVE_PATH);
end