function [feature,label,idx] = getFeatureAndLabel(RealData,ImagData,DataLabel,TargetLabel)
% This function is to
%   1. transform the received OFDM packets to feature vectors for training
%      data collection;
%   2. collect the corresponding labels.
% Determine the feature vector dimensions
[NumSym,NumSC,~] = size(RealData);
DimFeatureVec = NumSym*NumSC*2;
% Find packets of the target label
idx = find(DataLabel == TargetLabel);
numPacket = length(idx);
% Data collection
RealPart = RealData(:,:,idx); 
RealPart = permute(RealPart,[2,1,3]); 
RealPart = reshape(RealPart,NumSC*NumSym,numPacket); 
ImagPart = ImagData(:,:,idx);
ImagPart = permute(ImagPart,[2,1,3]); 
ImagPart = reshape(ImagPart,NumSC*NumSym,numPacket); 
% Feature vector
feature = zeros(DimFeatureVec,numPacket);
feature(1:2:end,:) = RealPart;
feature(2:2:end,:) = ImagPart;
% Label collection
label = TargetLabel*ones(1,numPacket);
end



