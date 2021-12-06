function [similarity] = multivarLPS(ensemble,testseries,trainseries)
    % multivariateLPS  Compute similarity between test and train series
    % returns a distance matrix of size number of test series by number of
    % train series
    %dbstop 78;
    noftrain=length(trainseries);
    noftest=length(testseries);
    tlenTrain=cell2mat(cellfun(@(x) size(x,2), trainseries, 'UniformOutput', false));
    tlenTest=cell2mat(cellfun(@(x) size(x,2), testseries, 'UniformOutput', false));
    
    observations=zeros(sum(tlenTrain),ensemble.mdim);
    diffobservations=zeros(sum(tlenTrain-1),ensemble.mdim);
    observationsTest=zeros(sum(tlenTest),ensemble.mdim);
    diffobservationsTest=zeros(sum(tlenTest-1),ensemble.mdim);
    for m=1:ensemble.mdim
        observations(:,m)=cell2mat(cellfun(@(x) x(m,:), trainseries, 'UniformOutput', false));
        diffobservations(:,m)=cell2mat(cellfun(@(x) diff(x(m,:)), trainseries, 'UniformOutput', false));
        observationsTest(:,m)=cell2mat(cellfun(@(x) x(m,:), testseries, 'UniformOutput', false));
        diffobservationsTest(:,m)=cell2mat(cellfun(@(x) diff(x(m,:)), testseries, 'UniformOutput', false));
    end
 
    startIndices=zeros(0,noftrain+1);
    startIndices(2:(noftrain+1))=tlenTrain;
    startIndices=cumsum(startIndices);
    startIndicesdiff=zeros(0,noftrain+1);
    startIndicesdiff(2:(noftrain+1))=tlenTrain-1;
    startIndicesdiff=cumsum(startIndicesdiff);
    obsSeries=cell2mat(arrayfun(@(x) repmat(x,1,tlenTrain(x)), 1:noftrain, 'UniformOutput', false));
      
    startIndicesTest=zeros(0,noftest+1);
    startIndicesTest(2:(noftest+1))=tlenTest;
    startIndicesTest=cumsum(startIndicesTest);
    startIndicesdiffTest=zeros(0,noftest+1);
    startIndicesdiffTest(2:(noftest+1))=tlenTest-1;
    startIndicesdiffTest=cumsum(startIndicesdiffTest);
    obsSeriesTest=cell2mat(arrayfun(@(x) repmat(x,1,tlenTest(x)), 1:noftest, 'UniformOutput', false));
   
    ind=1:(2*ensemble.nsegment*ensemble.mdim);
    similarity=zeros(noftrain,noftest);
    for i=1:ensemble.ntree
        % definition of matrices to store training and test segments
        individualSegmentLens=floor(ensemble.segfrac(i)*tlenTrain);
        individualSegmentLensTest=floor(ensemble.segfrac(i)*tlenTest);
        individualSegmentLens(individualSegmentLens<2)=2;
        individualSegmentLensTest(individualSegmentLens<2)=2;
     
        trainsegments=zeros(sum(individualSegmentLens),2*ensemble.nsegment*ensemble.mdim);
        testsegments=zeros(sum(individualSegmentLensTest),2*ensemble.nsegment*ensemble.mdim);

        count=0;
        for k=1:ensemble.nsegment;
            segIndices=cell2mat(arrayfun(@(x) (ensemble.stx(i,k)+startIndices(x)):(ensemble.stx(i,k)+startIndices(x)+individualSegmentLens(x)-1), ...
                            1:noftrain, 'UniformOutput', false));
            segIndicesdiff=cell2mat(arrayfun(@(x) (ensemble.stxdiff(i,k)+startIndicesdiff(x)):(ensemble.stxdiff(i,k)+startIndicesdiff(x)+individualSegmentLens(x)-1), ...
                            1:noftrain, 'UniformOutput', false));
            segIndicesTest=cell2mat(arrayfun(@(x) (ensemble.stx(i,k)+startIndicesTest(x)):(ensemble.stx(i,k)+startIndicesTest(x)+individualSegmentLensTest(x)-1), ...
                            1:noftest, 'UniformOutput', false));
            segIndicesdiffTest=cell2mat(arrayfun(@(x) (ensemble.stxdiff(i,k)+startIndicesdiffTest(x)):(ensemble.stxdiff(i,k)+startIndicesdiffTest(x)+individualSegmentLensTest(x)-1), ...
                            1:noftest, 'UniformOutput', false));
            for m=1:ensemble.mdim; 
                count=count+1;
                trainsegments(:,count)=observations(segIndices,m); 
                testsegments(:,count)=observationsTest(segIndicesTest,m); 
                count=count+1;
                trainsegments(:,count)=diffobservations(segIndicesdiff,m); 
                testsegments(:,count)=diffobservationsTest(segIndicesdiffTest,m); 
            end
        end
        
        trindex=obsSeries(segIndices);
        tstindex=obsSeriesTest(segIndicesTest);
        
        [yfit,nodes]=eval(ensemble.trees{i},trainsegments(:,ind~=ensemble.target(i)));
        [codetr,chi2,p,trlabels]=crosstab(trindex,nodes);

        [yfit,nodes]=eval(ensemble.trees{i},testsegments(:,ind~=ensemble.target(i)));
        [codetst,chi2,p,tstlabels]=crosstab(tstindex,nodes);
        if(size(codetr,2)==size(codetst,2)) %needs work, for now skipping trees for not balanced representations
            minmatch = @(XI,XJ,W) (bsxfun(@min,XI,XJ)*W);
            similarity = similarity + pdist2(codetr,codetst,@(XI,XJ) minmatch(XI,XJ,ones(size(codetr,2),1)));
        end
    end
end