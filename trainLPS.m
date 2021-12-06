function [ensemble] = trainLPS(series,ntree,nsegment,minseg,maxseg,nvartosample)
    % trainmultivariateLPS  Train ensemble of regression trees for
    % representation learning
    % ensemble=trainmultivariateLPS(series,diffseries,ntree,nsegment)
    %
    % series: cell structure that stores individual (univariate) series of 
    % multivariate time series (MTS)
    %
    % diffseries: cell structure that stores the consecutive differences of
    % each univariate time series of MTS
    %
    % ntree: number of trees in the ensemble
    % nsegment: number of segments to be considered for each series and
    % diffseries
    % dbstop 55;
    nofseries=length(series);
    tlen=cell2mat(cellfun(@(x) size(x,2), series, 'UniformOutput', false));
    mdim=size(series{1},1);
    
    observations=zeros(sum(tlen),mdim);
    diffobservations=zeros(sum(tlen-1),mdim);
    for m=1:mdim
        observations(:,m)=cell2mat(cellfun(@(x) x(m,:), series, 'UniformOutput', false));
        diffobservations(:,m)=cell2mat(cellfun(@(x) diff(x(m,:)), series, 'UniformOutput', false));
    end
    
    startIndices=zeros(0,nofseries+1);
    startIndices(2:(nofseries+1))=tlen;
    startIndices=cumsum(startIndices);
    startIndicesdiff=zeros(0,nofseries+1);
    startIndicesdiff(2:(nofseries+1))=tlen-1;
    startIndicesdiff=cumsum(startIndicesdiff);

    minlen=min(tlen);
    segfrac=minseg+(maxseg-minseg)*rand(ntree,1);
   
    ind=1:(2*nsegment*mdim);
    target=randsample(ind,ntree,true);

    ensemble=struct;
    ensemble.nsegment=nsegment;
    ensemble.ntree=ntree;
    ensemble.segfrac=segfrac;
    ensemble.minlen=minlen;
    ensemble.mdim=mdim;
    ensemble.target=target;
    ensemble.stx=zeros(ntree,nsegment);
    ensemble.stxdiff=zeros(ntree,nsegment);

    for i=1:ntree;
        individualSegmentLens=floor(segfrac(i)*tlen);
        individualSegmentLens(individualSegmentLens<2)=2;
        maxsegmentlen=min(minlen-2,min(individualSegmentLens));
        stx=randsample(minlen-maxsegmentlen,nsegment,true);             
        stxdiff=randsample(minlen-maxsegmentlen-1,nsegment,true);   
        segments=zeros(sum(individualSegmentLens),2*nsegment*mdim);
        count=0;
        for k=1:nsegment;
            segIndices=cell2mat(arrayfun(@(x) (stx(k)+startIndices(x)):(stx(k)+startIndices(x)+individualSegmentLens(x)-1), ...
                            1:nofseries, 'UniformOutput', false));
            segIndicesdiff=cell2mat(arrayfun(@(x) (stxdiff(k)+startIndicesdiff(x)):(stxdiff(k)+startIndicesdiff(x)+individualSegmentLens(x)-1), ...
                            1:nofseries, 'UniformOutput', false));
            for m=1:mdim; 
                count=count+1;
                segments(:,count)=observations(segIndices,m);  
                count=count+1;
                segments(:,count)=diffobservations(segIndicesdiff,m); 
            end
        end
        tree = classregtree(segments(:,ind~=target(i)),segments(:,target(i)),'method','regression', ...
            'prune','off','minleaf',floor(0.01*size(segments,1))+1,'nvartosample',nvartosample);
        ensemble.trees{i}=tree;
        ensemble.stx(i,:)=stx;
        ensemble.stxdiff(i,:)=stxdiff;
    end

end

