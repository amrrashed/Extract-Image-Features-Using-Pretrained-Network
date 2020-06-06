clc;close all;clear
%%cross validation
%load images
%D:\CIT project\datasets\Dataset_BUSI\old_with_BUSI
%D:\CIT project\datasets\Dataset_BUSI\ultrasonic 224
digitDatasetPath = fullfile('D:\CIT project\datasets\ultrasonic images\us-dataset\modified 224');
 imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Determine the split up
total_split=countEachLabel(imds)
% Number of Images
num_images=length(imds.Labels);
[imdsTrain,imdsTest] = splitEachLabel(imds,0.8,'randomized');
 numClasses = numel(categories(imdsTrain.Labels));
%    % Data Augumentation
%     augmenter = imageDataAugmenter( ...
%         'RandRotation',[-5 5],'RandXReflection',1,...
%         'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
%     
%    % Resizing all training images to [224 224] for ResNet architecture
    %imds = augmentedImageDatastore([227 227],imds);
%  
    net =vgg19;
    analyzeNetwork(net)
    layer = 'fc7';
featuresTrain = activations(net,imdsTrain,layer,'OutputAs','rows','ExecutionEnvironment','cpu');
featuresTest = activations(net,imdsTest,layer,'OutputAs','rows','ExecutionEnvironment','cpu');
%featuresall = activations(net,imds,layer,'OutputAs','rows','ExecutionEnvironment','cpu');
whos featuresTrain
 YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
%Yall = imds.Labels;
%merge all features in one file
%all={featuresall,Yall};
%xlswrite('myFile.xlsx',featuresall,1)
%xlswrite('myFile.xlsx',Yall,1,'SS')
%Fit Image Classifier
%Use the features extracted from the training images as predictor variables and fit a multiclass support vector machine (SVM) using fitcecoc (Statistics and Machine Learning Toolbox).
classifier = fitcecoc(featuresTrain,YTrain);
%Classify Test Images
%Classify the test images using the trained SVM model using the features extracted from the test images.

YPred = predict(classifier,featuresTest);
%Display four sample test images with their predicted labels.

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end
    accuracy = mean(YPred == YTest)