function refSignal=ref_signal_nh(f,fs,phase,tlen,num_of_harmonics)
%% sine-cosine reference
p=tlen;
% fs=250;
% TP=1/fs:1/fs:p/fs;
TP=0:1/fs:p/fs-1/fs;
refSignal=[];
for h=1:num_of_harmonics
    Sinh1=sin(2*pi*h*f*TP+h*phase);
    Cosh1=cos(2*pi*h*f*TP+h*phase);
    refSignal=[refSignal;Sinh1;Cosh1;];
end
