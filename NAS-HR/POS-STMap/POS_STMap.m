function POS_STMap(video_path, landmark_path, dst_path, fps, lmk_num)
%POS_STMap  Reads .dat landmarks, calls getROI_signal, forms STMap, applies POS.

if nargin < 5
    lmk_num = 468;  % default to 468 for MediaPipe
end

ap = 0.3;
as = 10;
wp = 0.6;
ws = 4;
wpp = wp/(fps/2);
wss = ws/(fps/2);
[n, wn] = buttord(wpp, wss, ap, as);
[b, a] = butter(n, wn);

% Ensure dst_path is a char, then create folder if needed
if iscell(dst_path)
    dst_path = cell2mat(dst_path);
end
if ~exist(dst_path, 'dir')
    mkdir(dst_path);
end

obj = VideoReader(video_path);
numFrames = obj.NumFrames;
signal = [];

for k = 1:numFrames
    dat_file = fullfile(landmark_path, sprintf('landmarks%d.dat', k));
    fid = fopen(dat_file, 'r');
    if fid > 0
        % Read 2*lmk_num int32 values => 936 if lmk_num=468
        landmarks = fread(fid, 2*lmk_num, 'int32');
        fclose(fid);
    else
        landmarks = zeros(2*lmk_num, 1);
    end
    
    frame = read(obj, k);
    
    % Extract 1x45 ROI signals
    s = getROI_signal(frame, landmarks);
    % Concatenate row-wise => [numFrames x 45]
    signal = [signal; s];
end

[rows, totalCols] = size(signal);
if totalCols < 45
    warning('Signal does not have the expected 45 columns. Check getROI_signal.');
end

% Reshape => Combine_channel(15, rows, 3)
Combine_channel = zeros(15, rows, 3);
Combine_channel(:,:,1) = signal(:, 1:15)';
Combine_channel(:,:,2) = signal(:, 16:30)';
Combine_channel(:,:,3) = signal(:, 31:45)';

% Apply POS on the 3rd channel
for n = 1:15
    Combine_channel(n,:,3) = POS(squeeze(Combine_channel(n,:,:)), fps);
end

save_map_fullTime(dst_path, Combine_channel);
end
