function SER = getSymbolDetection(ReceivedData,EstChan,Mod_Constellation,Label,DataLabel)
% This function is to calculate the symbol error rate from the equalized
% symbols based on hard desicion. 
EstSym = squeeze(ReceivedData)./EstChan;
% Hard decision
DecSym = sign(real(EstSym))+1j*sign(imag(EstSym));
DecLabel = zeros(size(DecSym));
for c = 1:length(Mod_Constellation)
    DecLabel(logical(DecSym == Mod_Constellation(c))) = Label(c);
end
SER = 1-sum(DecLabel == DataLabel)/length(EstSym);
end