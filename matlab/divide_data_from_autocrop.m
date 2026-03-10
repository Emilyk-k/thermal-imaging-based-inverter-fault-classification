sourceRoot = fullfile('..', '..','..', 'dataset', 'inverter_40_faults_duration_270s_20260114_124337', 'autocrop');
trainRoot  = fullfile('data', 'train_single_speed_40_classes_transient');
testRoot   = fullfile('data', 'test_single_speed_40_classes_transient');
trainRatio = 0.7;
steadyStateOnly = false;

faultDirs = dir(fullfile(sourceRoot, 'fault_*_speed_*'));
faultDirs = faultDirs([faultDirs.isdir]); 

if steadyStateOnly
    subDirs = {'steady_state'};
    fprintf('Mode: Using STEADY_STATE data only.\n');
else
    subDirs = {'steady_state', 'transient_state'};
    fprintf('Mode: Using BOTH steady and transient data.\n');
end

fprintf('Starting data distribution...\n');

for i = 1:length(faultDirs)
    sourceDirName = faultDirs(i).name;
    
    parts = split(sourceDirName, '_');
    className = strjoin(parts(1:2), '_'); 
    
    fprintf('Processing: %s -> Mapping to: %s\n', sourceDirName, className);
    
    classTrainPath = fullfile(trainRoot, className);
    classTestPath  = fullfile(testRoot, className);
    
    if ~exist(classTrainPath, 'dir'), mkdir(classTrainPath); end
    if ~exist(classTestPath, 'dir'), mkdir(classTestPath); end
    
    allFiles = [];
    for s = 1:length(subDirs)
        currentSubPath = fullfile(sourceRoot, sourceDirName, subDirs{s});
        if ~exist(currentSubPath, 'dir'), continue; end
        
        filesInSub = dir(fullfile(currentSubPath, '*.png'));
        
        for f = 1:length(filesInSub)
            filesInSub(f).fullPath = fullfile(filesInSub(f).folder, filesInSub(f).name);
            filesInSub(f).destName = [sourceDirName '_' subDirs{s} '_' filesInSub(f).name];
        end
        allFiles = [allFiles; filesInSub];
    end
    
    numFiles = length(allFiles);
    if numFiles == 0
        warning('No files found for %s. Skipping.', sourceDirName);
        continue;
    end
    
    shuffledIdx = randperm(numFiles);
    splitPoint = floor(numFiles * trainRatio);
    
    for j = 1:numFiles
        if j <= splitPoint
            destFolder = classTrainPath;
        else
            destFolder = classTestPath;
        end
        
        destFile = fullfile(destFolder, allFiles(shuffledIdx(j)).destName);
        copyfile(allFiles(shuffledIdx(j)).fullPath, destFile);
    end
end

fprintf('Process complete. Data is organized in %s and %s\n', trainRoot, testRoot);