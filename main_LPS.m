clear all;
clc;
load('data/UWave.mat');

trainOnly=false;
ntree=200;  % number of trees
nsegment=10; % number of segments to use for each tree

error_rate=zeros(10,1);
train_time=zeros(10,1);
test_time=zeros(10,1);

%read data of individual axis
[train trainclass test testclass]=preprocess(mts,trainOnly);
noftest=length(testclass);
noftrain=length(trainclass);

%replicate 10 times for just experimental purposes
for repl=1:10;
    tic;
    [ensemble]=trainLPS(train,ntree,nsegment,0.1,0.9,5);
    elapsedtr=toc;
    fprintf('Replication %d, Training time: %.3f seconds\n',repl,elapsedtr);
    tic;
    similarity=multivarLPS(ensemble,test,train);
    if(trainOnly)
        similarity(1:noftrain+1:end)=0;
    end
    [C,I] = max(similarity);    % find training series with minimum distance
    fitted = trainclass(I);     % 1NN classification
    elapsedtst=toc;
    
    error_rate(repl)=1-sum(fitted==testclass)/noftest;
    train_time(repl)=elapsedtr;
    test_time(repl)=elapsedtst/noftest;
    
    fprintf('Test time: %.4f seconds\n',elapsedtst/noftest);
    fprintf('Test error rate %.3f\n',error_rate(repl));
end

fprintf('Average error rate %.3f, training time %.3f, test time %.3f\n', ...
    mean(error_rate),mean(train_time),mean(test_time));

