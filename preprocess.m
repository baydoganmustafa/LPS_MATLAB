function [train, trainclass, test, testclass]=preprocess(dat,trainOnly)
    train=dat.train;
    trainclass=dat.trainlabels;
    
    if(trainOnly)
        test=dat.train;
        testclass=dat.trainlabels;       
    else
        test=dat.test;
        testclass=dat.testlabels;
    end
end