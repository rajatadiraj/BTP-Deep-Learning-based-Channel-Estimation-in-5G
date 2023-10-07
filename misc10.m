%% Testing
% 
% This script
%   1. generates testing data for each SNR point;
%   2. calculates the symbol error rate (SER) based on deep learning (DL), 
%   least square (LS) and minimum mean square error (MMSE).

%% Clear workspace
clc;
clear all;
close all;

%% Load common parameters and the trained NN

load('SimParametersPilot64.mat'); 
%loads a structure containing fields like FixedPilot, Label, LengthCP,
%Mod_Constellation, NumDataSym, NumPilotSym, NumSC, h, idxSC
load('TrainedNetPilot64.mat');
%loads a strcuture containing MiniBatchSize and Net. Net contains Layers,
%Input names and Output names.

%% Other simulation parameters

NumPilot = length(FixedPilot);     %64 in our case
PilotSpacing = NumSC/NumPilot;     %NumSC=Number of OFDM subcarriers=64 in our case
NumOFDMsym = NumPilotSym+NumDataSym; % 1+1=2
NumClass = length(Label);          %label=[1,2,3,4] so NumClass=4
NumPath = length(h);               %NumPath=20 in our case

% Load pre-calculated channel autocorrelation matrix for MMSE estimation
% This autocorrelation matrix is calculated in advance using the 3GPP
% channel model, which can be replaced accordingly.
load('RHH.mat');        %64x64 values
%RHH= auto-covariance of the channel frequency response H
%RHY= cross-covariance matrix of channel transfer function and received s/g
%RYY= auto-covariance matrix of the received signal.
%Since the LS algorithm is not highly accurate due to noise interference,
%the MMSE channel estimation is proposed It uses the second-order
%statistical properties of the channel to reduce the MSE and greatly
%improves the accuracy of the channel estimation

%% SNR range

Es_N0_dB = 0:2:20; % Es/N0 in dB
Es_N0 = 10.^(Es_N0_dB./10); % linear Es/N0
N0 = 1./Es_N0;
NoiseVar = N0./2;

%% Testing data size

NumPacket = 10000; % Number of packets simulated per iteration

%% Simulation

% Same pilot sequences used in training and testing stages
FixedPilotAll = repmat(FixedPilot,1,1,NumPacket);  %repmat for Repeat copies of array repeats
%FixedPilot 10000 times.

%Monte Carlo simulation is a computerized mathematical technique that allows people to account for risk in 
%quantitative analysis and decision making.
%During a Monte Carlo simulation, values are sampled at random from the input probability distributions. 
%Each set of samples is called an iteration, and the resulting outcome from that sample is recorded. 
%Monte Carlo simulation does this hundreds or thousands of times, and the result is a probability 
%distribution of possible outcomes. In this way, Monte Carlo simulation provides a much more comprehensive 
%view of what may happen. It tells you not only what could happen, but how likely it is to happen.
% Number of Monte-Carlo iterations
NumIter = 1;

% Initialize error rate vectors
%SER_DL = zeros(length(NoiseVar),NumIter);
SER_LS = zeros(length(NoiseVar),NumIter);  
SER_MMSE = zeros(length(NoiseVar),NumIter);

for i = 1:NumIter
    
    for snr = 1:length(NoiseVar)
        
        %% 1. Testing data generation
        
        noiseVar = NoiseVar(snr);
                
        % OFDM pilot symbol (can be interleaved with random data symbols)
        PilotSym = 1/sqrt(2)*complex(sign(rand(NumPilotSym,NumSC,NumPacket)-0.5),sign(rand(NumPilotSym,NumSC,NumPacket)-0.5)); 
        PilotSym(1:PilotSpacing:end) = FixedPilotAll;
    
        % OFDM data symbol
        DataSym = 1/sqrt(2)*complex(sign(rand(NumDataSym,NumSC,NumPacket)-0.5),sign(rand(NumDataSym,NumSC,NumPacket)-0.5)); 
    
        % Transmitted OFDM frame
        TransmittedPacket = [PilotSym;DataSym];
        
        % Received OFDM frame
        ReceivedPacket = genTransmissionReceptionOFDM(TransmittedPacket,LengthCP,h,noiseVar); 
        %LengthCP=OFDM cyclic prefix length
        
        % Collect the data labels for the selected subcarrier
        DataLabel = zeros(size(DataSym(:,idxSC,:)));
        for c = 1:NumClass
            DataLabel(logical(DataSym(:,idxSC,:) == 1/sqrt(2)*Mod_Constellation(c))) = Label(c);
        %L = logical(A) converts A into an array of logical values. Any nonzero element of A is converted to 
        %logical 1 (true) and zeros are converted to logical 0 (false). 
        end
        DataLabel = squeeze(DataLabel); 
        %B = squeeze(A) returns an array with the same elements as the input array A, but with dimensions 
        %of length 1 removed. For example, if A is a 3-by-1-by-1-by-2 array, then squeeze(A) returns a 
        %3-by-2 matrix.
        % Testing data collection
     %   XTest = cell(NumPacket,1);
        %A cell array is a data type with indexed data containers called cells, where each cell can contain 
        %any type of data. Cell arrays commonly contain either lists of text, combinations of text and numbers,
        %or numeric arrays of different sizes.
     %   YTest = zeros(NumPacket,1);       
     %   for c = 1:NumClass
     %       [feature,label,idx] = getFeatureAndLabel(real(ReceivedPacket),imag(ReceivedPacket),DataLabel,Label(c));
     %       featureVec = mat2cell(feature,size(feature,1),ones(1,size(feature,2))); 
            %mat2cell- Convert array to cell array whose cells contain subarrays
     %       XTest(idx) = featureVec;
     %       YTest(idx) = label;
     %   end
     %   YTest = categorical(YTest);
        %Array that contains values assigned to categories
        %categorical is a data type that assigns values to a finite set of discrete categories, such as High,
        %Med, and Low. These categories can have a mathematical ordering that you specify, such as 
        %High > Med > Low, but it is not required. 
        % B = categorical(A) creates a categorical array from the array A. The categories of B are the sorted 
        %unique values from A.
        %% 2. DL detection
        
       % YPred = classify(Net,XTest,'MiniBatchSize',MiniBatchSize);
        %SER_DL(snr,i) = 1-sum(YPred == YTest)/NumPacket;
        
        %% 3. LS & MMSE detection
        
        % Channel estimation
        wrapper = @(x,y) performChanEstimation(x,y,RHH,noiseVar,NumPilot,NumSC,NumPath,idxSC);
        ReceivedPilot = mat2cell(ReceivedPacket(1,:,:),1,NumSC,ones(1,NumPacket));
        PilotSeq = mat2cell(FixedPilotAll,1,NumPilot,ones(1,NumPacket));
        [EstChanLS,EstChanMMSE] = cellfun(wrapper,ReceivedPilot,PilotSeq,'UniformOutput',false);
        %cellfun-Apply function to each cell in cell array
        %A = cellfun(func,C) applies the function func to the contents of each cell of cell array C, 
        %one cell at a time. cellfun then concatenates the outputs from func into the output array A, 
        %so that for the ith element of C, A(i) = func(C{i}).
        EstChanLS = cell2mat(squeeze(EstChanLS));
        EstChanMMSE = cell2mat(squeeze(EstChanMMSE));
        %A = cellfun(func,C) applies the function func to the contents of each cell of cell array C, 
        %one cell at a time. cellfun then concatenates the outputs from func into the output array A, 
        %so that for the ith element of C, A(i) = func(C{i}).
        %The elements of the cell array must all contain the same data type, and the resulting array is 
        %of that data type.
        
        % Symbol detection
        SER_LS(snr,i) = getSymbolDetection(ReceivedPacket(2,idxSC,:),EstChanLS,Mod_Constellation,Label,DataLabel);
        SER_MMSE(snr,i) = getSymbolDetection(ReceivedPacket(2,idxSC,:),EstChanMMSE,Mod_Constellation,Label,DataLabel);
        
    end
    
end

%SER_DL = mean(SER_DL,2).';
SER_LS = mean(SER_LS,2).';
SER_MMSE = mean(SER_MMSE,2).';


figure();
%semilogy(Es_N0_dB,SER_DL,'r-o','LineWidth',2,'MarkerSize',10);hold on;
semilogy(Es_N0_dB,SER_LS,'b-o','LineWidth',2,'MarkerSize',10);hold on;
semilogy(Es_N0_dB,SER_MMSE,'k-o','LineWidth',2,'MarkerSize',10);hold off;
legend('Least square (LS)','Minimum mean square error (MMSE)');
xlabel('Es/N0 (dB)');
ylabel('Symbol error rate (SER)');

%%

% function [EstChanLS,EstChanMMSE] = performChanEstimation(ReceivedData,PilotSeq,RHH,NoiseVar,NumPilot,NumSC,NumPath,idxSC)
% % This function is to perform LS and MMSE channel estimations using pilot
% % symbols, second-order statistics of the channel and noise variance [1].
% 
% % [1] O. Edfors, M. Sandell, J. -. van de Beek, S. K. Wilson and 
% % P. Ola Borjesson, "OFDM channel estimation by singular value 
% % decomposition," VTC, Atlanta, GA, USA, 1996, pp. 923-927 vol.2.
% 
% 
% PilotSpacing = NumSC/NumPilot;
% 
% %%%%%%%%%%%%%%% LS estimation with interpolation %%%%%%%%%%%%%%%%%%%%%%%%%
% 
% H_LS = ReceivedData(1:PilotSpacing:NumSC)./PilotSeq;
% H_LS_interp = interp1(1:PilotSpacing:NumSC,H_LS,1:NumSC,'linear','extrap');
% H_LS_interp = H_LS_interp.';
% 
% %%%%%%%%%%%%%%%% MMSE estimation based on LS %%%%%%%%%%%%%%%%
% 
% [U,D,V] = svd(RHH);
% d = diag(D);
% 
% InvertValue = zeros(NumSC,1);
% if NumPilot >= NumPath
%     
%     InvertValue(1:NumPilot) = d(1:NumPilot)./(d(1:NumPilot)+NoiseVar);
%     
% else
%     
%     InvertValue(1:NumPath) = d(1:NumPath)./(d(1:NumPath)+NoiseVar);
%     
% end
% 
% H_MMSE = U*diag(InvertValue)*V'*H_LS_interp;
% 
% %%%%%%%%%%%%%%% Channel coefficient on the selected subcarrier %%%%%%%%%%%
% 
% EstChanLS = H_LS_interp(idxSC);
% EstChanMMSE = H_MMSE(idxSC);
% 
% end
% 
% %% 
% 
% function SER = getSymbolDetection(ReceivedData,EstChan,Mod_Constellation,Label,DataLabel)
% % This function is to calculate the symbol error rate from the equalized
% % symbols based on hard desicion. 
% 
% EstSym = squeeze(ReceivedData)./EstChan;
% 
% % Hard decision
% DecSym = sign(real(EstSym))+1j*sign(imag(EstSym));
% DecLabel = zeros(size(DecSym));
% for c = 1:length(Mod_Constellation)
%     DecLabel(logical(DecSym == Mod_Constellation(c))) = Label(c);
% end
% 
% SER = 1-sum(DecLabel == DataLabel)/length(EstSym);
% 
% end






