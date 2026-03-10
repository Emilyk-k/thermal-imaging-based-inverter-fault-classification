function detection()
    modelPath = config.MODEL_SAVE_PATH;
    fprintf('Loading model from: %s\n', modelPath);
    
    load(modelPath, 'trainedNet');
    model = trainedNet;
    
    fprintf('Model loaded successfully\n');
    
    try
        testImds = imageDatastore(config.TEST_DATA, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames', ...
        'FileExtensions', config.EXTENSION);
    
        testDatastore = augmentedImageDatastore([config.IMG_HEIGHT, config.IMG_WIDTH], ...
        testImds, ...
        'ColorPreprocessing', 'gray2rgb');
        trueLabels = testImds.Labels;
        
        fprintf('Evaluating model on test set...\n');
        startTime = tic;
        predictions = classify(model, testDatastore);
        trainingTime = toc(startTime);
        
        % % --- LIME ---
        % figure('Name', 'LIME Analysis');
        % numToItems = min(4, numel(testImds.Files)); 
        % 
        % for i = 1:numToItems
        %     img = readimage(testImds, i);
        %     [~, fileName, fileExt] = fileparts(testImds.Files{i});
        %     fullName = [fileName, fileExt];
        % 
        %     if size(img, 3) == 1
        %         img = cat(3, img, img, img);
        %     end
        % 
        %     imgResized = imresize(img, [config.IMG_HEIGHT, config.IMG_WIDTH]);
        % 
        %     [map, ~] = imageLIME(model, imgResized, predictions(i), ...
        %         'NumSamples', 500, ...
        %         'Segmentation', 'superpixels');
        % 
        %     subplot(1, numToItems, i);
        % 
        %     imshow(imgResized); 
        %     hold on;
        % 
        %     h = imagesc(map);
        %     set(h, 'AlphaData', 0.5); 
        % 
        %     colormap parula; 
        %     axis image off; 
        % 
        %     title({fullName; sprintf('Pred: %s', string(predictions(i)))}, ...
        %           'Interpreter', 'none', 'FontSize', 8);
        % end
        % % ---------

        % --- Grad-CAM ---
        featureLayer = 'efficientnet-b0|model|blocks_15|conv2d_1|Conv2D';
        reductionLayer = 'prediction_softmax';

        figure('Name', 'Grad-CAM Analysis');
        numToItems = min(4, numel(testImds.Files)); 

        for i = 1:numToItems
            img = readimage(testImds, i);

            [~, fileName, fileExt] = fileparts(testImds.Files{i});
            fullName = [fileName, fileExt];

            if size(img, 3) == 1
                img = cat(3, img, img, img);
            end

            imgResized = imresize(img, [config.IMG_HEIGHT, config.IMG_WIDTH]);

            scoreMap = gradCAM(model, imgResized, predictions(i), ...
                'FeatureLayer', featureLayer, ...
                'ReductionLayer', reductionLayer);

            subplot(1, numToItems, i);
            imshow(imgResized); 
            hold on;

            h = imagesc(scoreMap);
            set(h, 'AlphaData', 0.3); 

            colormap jet;
            axis image off; 

            title({fullName; sprintf('Pred: %s', string(predictions(i)))}, ...
                  'Interpreter', 'none', 'FontSize', 8);
        end
        % -----------------

        % % --- Occlusion Sensitivity Analysis ---
        % targetFiles = {'image_fault_01.png', 'image_fault_32.png', 'image_fault_33.png', 'image_healthy.png'};
        % figure('Name', 'Occlusion Sensitivity Analysis');
        % 
        % plotIdx = 1;
        % for i = 1:numel(targetFiles)
        %     idx = find(contains(testImds.Files, targetFiles{i}), 1);
        % 
        %     if isempty(idx)
        %         fprintf('Warning: %s not found in test dataset.\n', targetFiles{i});
        %         continue;
        %     end
        % 
        %     img = readimage(testImds, idx);
        %     if size(img, 3) == 1
        %         img = cat(3, img, img, img);
        %     end
        %     imgResized = imresize(img, [config.IMG_HEIGHT, config.IMG_WIDTH]);
        % 
        %     map = occlusionSensitivity(model, imgResized, predictions(idx), ...
        %         'MaskSize', 20, ...
        %         'Stride', 10, ... 
        %         'ExecutionEnvironment', 'gpu', ...
        %         'MiniBatchSize', 32);
        % 
        %     subplot(1, numel(targetFiles), plotIdx);
        %     imshow(imgResized);
        %     hold on;
        % 
        %     h = imagesc(map);
        %     set(h, 'AlphaData', 0.5); 
        % 
        %     colormap jet;
        %     axis image off;
        %     title({targetFiles{i}; sprintf('Pred: %s', string(predictions(idx)))}, ...
        %           'Interpreter', 'none', 'FontSize', 8);
        % 
        %     plotIdx = plotIdx + 1;
        % end
        % % -----------------

        fprintf('Finished test\n');
        fprintf('Test time: %.2f min, %.2f s\n', trainingTime/60, trainingTime);
        
        accuracy = sum(predictions == trueLabels) / numel(trueLabels);
        
        cm = confusionmat(trueLabels, predictions);
        
        numClasses = size(cm, 1);
        precision = zeros(numClasses, 1);
        recall = zeros(numClasses, 1);
        
        for i = 1:numClasses
            precision(i) = cm(i,i) / sum(cm(:,i));
            recall(i) = cm(i,i) / sum(cm(i,:));
        end
        
        macroPrecision = mean(precision);
        macroRecall = mean(recall);
        
        fprintf('\n%s\n', repmat('=', 1, 30));
        fprintf('Results on test set\n');
        fprintf('%s\n', repmat('=', 1, 30));
        fprintf('Accuracy: %.4f\n', accuracy);
        fprintf('Macro Precision: %.4f\n', macroPrecision);
        fprintf('Macro Recall: %.4f\n', macroRecall);
        fprintf('Macro F1-Score: %.4f\n', 2 * (macroPrecision * macroRecall) / (macroPrecision + macroRecall));
        
        trueStrings = string(trueLabels);
        predStrings = string(predictions);
        
        cleanTrue = regexprep(trueStrings, '_', ' '); 
        cleanPreds = regexprep(predStrings, '_', ' ');
        
        figure;
        cm_plot = confusionchart(cleanTrue, cleanPreds);
        title('Confusion Matrix - Test Set');
        exportgraphics(gcf, 'results\confusion_matrix.png', 'Resolution', 300);
        
        results.accuracy = accuracy;
        results.precision = macroPrecision;
        results.recall = macroRecall;
        results.confusionMatrix = cm;
        
    catch e
        fprintf('Evaluation error: %s\n', e.message);
    end
end
