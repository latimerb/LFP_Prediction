clear all;  close all;
prelen = 50;            % Number of passed time samples (50 ms) for input
forward = 10;           % Number of time samples forward to predict (10 ms)
filt_xcld = 50;         % Duration at the edge of segments to be excluded (50 ms)
rng(111);

fs = 1000;
oscBand = [1,5;5,8;9,14;15,40;64,84;84,124];    % Filter bands
gamma_band = 5;
n_bands = size(oscBand,1);

%load(fullfile('dataset','subject4.mat'));
load('subject4.mat');
T = length(LFP);

%% Get start time points of each chunk
xcld = 200;             % Exclude 0 periods more than 200 ms
minLen = 15000;         % Minimum length for a segment
seg = getseg(LFP,xcld,minLen);  % get segment time
nseg = size(seg,2);     % number of segments
tseg = [seg(1,:)+1+filt_xcld;seg(2,:)-filt_xcld];   % valid time domain
Lseg = tseg(2,:)-tseg(1,:)+1;
L = sum(Lseg,2);
valid = false(1,T);     % mark for valid segment
for i = 1:nseg
    valid(seg(1,i)+1:seg(2,i)) = true;
end

%% filtfilt for ground truth. causal filter for training input.
ZS = (LFP-mean(LFP(valid)))/std(LFP(valid));    % z-score
[bFilt,aFilt] = butter(2,oscBand(gamma_band,:)/(fs/2));
ZS_causal = filter(bFilt,aFilt,ZS);
ZS_gamma = filtfilt(bFilt,aFilt,ZS);
disp(['Raw ZS std = ',num2str(std(ZS(valid)))]);
disp(['Causal std = ',num2str(std(ZS_causal(valid)))]);
disp(['Non-causal std = ',num2str(std(ZS_gamma(valid)))]);

%% training data
x = cell(1,nseg);
for i = 1:nseg
    x{i} = [ZS(tseg(1,i):tseg(2,i))';ZS_causal(tseg(1,i):tseg(2,i))';ZS_gamma(tseg(1,i):tseg(2,i))'];
end
traindata = cell2mat(x)';
csvwrite('train_forward_pred.csv',traindata);
dlmwrite('segment_length.csv',Lseg,'precision','%d');

