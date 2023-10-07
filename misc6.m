function ReceivedPacket = genTransmissionReceptionOFDM(TransmittedFrame,LengthCP,h,NoiseVar)
% This function is to model the transmission and reception process in OFDM systems. 
% Extract parameters
[NumSym,NumSC,NumPacket] = size(TransmittedFrame);
%% Transmitter
PhaseShift = exp(-1j*rand(1,NumPacket)*2*pi);
for p = 1:NumPacket
    % 1. IFFT
    x1 = ifft(TransmittedFrame(:,:,p),NumSC,2); 
    % 2. Inserting CP
    x1_CP = [x1(:,NumSC-LengthCP+1:end) x1]; 
    % 3. Parallel to serial transformation
    x2 = x1_CP.';
    x = x2(:);
    % 4. Channel filtering
    y_conv = conv(h*PhaseShift(p),x);
    y(:,p) = y_conv(1:length(x)); 
end           
%% Adding noise 
SeqLength = size(y,1);
% Calculate random noise in time domain 
NoiseF = sqrt(NoiseVar)/sqrt(2).*(randn(NumPacket,NumSC)+1j*randn(NumPacket,NumSC)); % Frequency-domain noise
NoiseT = sqrt(SeqLength)*sqrt(SeqLength/NumSC)*ifft(NoiseF,SeqLength,2); % Time-domain noise
% Adding noise
y = y+NoiseT.'; 
%% Receiver
ReceivedPacket = zeros(NumPacket,NumSym,NumSC); 
for p = 1:NumPacket
    % 1. Serial to parallem transformation
    y1 = reshape(y(:,p),NumSC+LengthCP,NumSym).'; 
    % 2. Removing CP
    y2 = y1(:,LengthCP+1:LengthCP+NumSC);
    % 3. FFT, # x NymSym x 64
    ReceivedPacket(p,:,:) = fft(y2,NumSC,2); % NumSym x 64
end 
ReceivedPacket = permute(ReceivedPacket,[2,3,1]); 
