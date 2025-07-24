% function save_map_fullTime(HR_train_path, SignalMap)
% 
% final_signal = SignalMap;
% img_path = strcat(HR_train_path, '\img_mvavg_full.png');
% channel_num = size(SignalMap,3);
% judge = mean(final_signal,1);   
% if ~isempty(find(judge(1,:,2) == 0))
%      a = 0;
% else 
%     final_signal1 = final_signal;
%     for idx = 1:size(final_signal,1)
%         for c = 1:channel_num
%             temp = final_signal(idx,:,c);
%             % temp = movmean(temp,3);
%             final_signal1(idx,:,c) = (temp - min(temp))/(max(temp) - min(temp))*255;
%         end
%     end
% 
%     final_signal1 = final_signal1(:,:,[1 2 3]);
%     img1 = final_signal1;
%     imwrite(uint8(img1), img_path);
% end
function save_map_fullTime(HR_train_path, SignalMap)
%save_map_fullTime  Writes "img_mvavg_full.png" from the 3D spatiotemporal data.

if ~exist(HR_train_path, 'dir')
    mkdir(HR_train_path);
end

img_path = fullfile(HR_train_path, 'img_mvavg_full.png');
final_signal = SignalMap;  % e.g. size = [15 x frames x 3]
channel_num = size(final_signal,3);

% Instead of skipping if channel2=0, we always save:
final_signal1 = zeros(size(final_signal));
for idx = 1:size(final_signal,1)
    for c = 1:channel_num
        temp = final_signal(idx,:,c);
        denom = max(temp) - min(temp);
        if denom < eps
            denom = eps;
        end
        final_signal1(idx,:,c) = (temp - min(temp)) / denom * 255;
    end
end

img1 = final_signal1(:,:,[1 2 3]);
imwrite(uint8(img1), img_path);
disp(['>> STMap saved to: ', img_path]);
end
