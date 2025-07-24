% 
% warning off
% source1 = {'s1','s2','s3'};
% source2 = {'source1','source2','source3'};
% % landmark path seeta face
% Lmk_root_path = 'D:\HR_landmarks\SDK_smooth';
% % vedio path
% video_root_path = 'H:\VIPL_HR';
% save_path ='D:\Data\VIPL_my15_YUV_LB\';
% % person index
% p_path = dir(Lmk_root_path);
% p_path = sort_nat({p_path.name});
% p_path = p_path(3:end);
% lmk_num = 81;
% count = 0;
% dizhen = 0;
% for p_index = 1:length(p_path)
%     p_now = strcat(Lmk_root_path,'/', cell2mat(p_path(p_index)));
%     v_path = dir(p_now);
%     v_path = sort_nat({v_path.name});
% 	v_path = v_path(3:end);
%     % vedio
%     p_now_vedio = strcat(video_root_path,'/', cell2mat(p_path(p_index)));
%     v_path_vedio = dir(p_now_vedio);
%     v_path_vedio = sort_nat({v_path_vedio.name});
% 	v_path_vedio = v_path_vedio(3:end);  
%     v_length= min(length(v_path),length(v_path_vedio));
% 
%     for v_index = 1:v_length
%         v_now = strcat(p_now,'/', cell2mat(v_path(v_index)));
%         s_path = dir(v_now);
%         s_path = sort_nat({s_path.name});
%         s_path = s_path(3:end);
%         % vedio
%         v_now_vedio = strcat(p_now_vedio,'/', cell2mat(v_path_vedio(v_index)));
%         s_path_vedio = dir(v_now_vedio);
%         s_path_vedio = sort_nat({s_path_vedio.name});
%         s_path_vedio = s_path_vedio(3:end);
%         % ��֤s��ͬ
%         falg1 = zeros(3,1);
%         falg2 = zeros(3,1);
%         for source_kind = 1:3
%            for pat = 1:length(s_path)
%                if(cell2mat(source1(source_kind))==cell2mat(s_path(pat)))
%                   falg1(source_kind) = 1;
%                end
%            end
%            for pat = 1:length(s_path_vedio)
%                if(cell2mat(source2(source_kind))==cell2mat(s_path_vedio(pat)))
%                   falg2(source_kind) = 1;
%                end
%            end 
%         end
%         s_real = find(falg1==falg2&falg1==1);
%         for s_index = 1:length(s_real)
%             s_now = strcat(v_now,'/', cell2mat(source1(s_real(s_index))));
%             % vedio
%             s_now_vedio = strcat(v_now_vedio,'/', cell2mat(source2(s_real(s_index))));
%             % now path
%             landmark_path = strcat(s_now,'/', 'face_landmarks');
%             vidio_path = strcat(s_now_vedio,'/', 'video.avi');
%             HR_path = strcat(s_now_vedio,'/', 'gt_HR.csv');
%             SpO2_path = strcat(s_now_vedio,'/', 'gt_SpO2.csv');
%             wave_path = strcat(s_now_vedio,'/', 'wave.csv');
%             % vidio
%             obj = VideoReader(vidio_path);
%             numFrames = obj.NumberOfFrames;
%             numlandmarks = length(dir(landmark_path))-2;
%             if numFrames==numlandmarks && numFrames>520
%                 count = count+1
%                 signal = [];
%                 % get HR/SpO2/
%                 HR = csvread(HR_path,1,0);
%                 SpO2 = csvread(SpO2_path,1,0);
%                 wave = csvread(wave_path,1,0);
%                 % ֡��
%                 fps = (numFrames/length(HR));
%                 if (fps > 15)
%                     % save path
%                     dst_path = strcat(save_path,p_path(p_index),v_path(v_index),s_path(s_index));
%                     % save wave as pic
%                     wave2Pic(wave,numFrames,dst_path);
%                     % save STmap as pic
%                     POS_STMap(vidio_path, landmark_path, dst_path, fps);
%                     % save HR,SpO2,fps
%                     % spline HR/SPO2
%                     HR_size = size(HR);
%                     SpO2_size = size(SpO2);
%                     w_x = linspace(0,100,HR_size(1));
%                     w_x1 = linspace(0,100,SpO2_size(1));
%                     d_x = linspace(0,100,numFrames-1);
%                     HR = spline(w_x,HR,d_x);
%                     SpO2 = spline(w_x1,SpO2,d_x);
%                     fps_path = strcat(cell2mat(dst_path), '/fps.mat');    
%                     HR_path = strcat(cell2mat(dst_path), '/HR.mat'); 
%                     SpO2_path = strcat(cell2mat(dst_path), '/SpO2.mat');
%                     eval(['save ', fps_path, ' fps']);
%                     eval(['save ', HR_path, ' HR']);
%                     eval(['save ', SpO2_path, ' SpO2']);                 
%                 end
%             else
% %                 numFrames
% %                 numlandmarks
%             end
%         end
%     end
% end

% function VIPL_Processing()
%     disp('>> Starting VIPL_Processing.m...');
% 
%     % We only expect "s1" for landmarks and "source1" for videos:
%     source1 = {'s1'};  
%     source2 = {'source1'}; 
% 
%     % Paths (adjust to your actual folder layout):
%     Lmk_root_path   = 'C:\Users\User\Desktop\Backup Lenovo\D Drive\Testing FYP\NAS-HR\Landmarks';       % Path to Landmarks
%     video_root_path = 'C:\Users\User\Desktop\Backup Lenovo\D Drive\Testing FYP\NAS-HR\video_root_path'; % Path to Videos
%     save_path       = 'C:\Users\User\Desktop\Backup Lenovo\D Drive\Testing FYP\NAS-HR\Output';          % Output Folder
% 
%     disp('>> Checking Landmarks folder structure...');
%     all_files = dir(Lmk_root_path);
%     disp(['>> Raw all_files struct: ', num2str(length(all_files)), ' entries found.']);
%     all_names = {all_files.name}; 
%     p_path = all_names([all_files.isdir]); 
%     p_path = p_path(~ismember(p_path, {'.', '..'}));  % remove '.' and '..'
%     disp(['>> Found persons in dataset: ', strjoin(p_path, ', ')]);
% 
%     lmk_num = 468;   % MediaPipe landmarks
%     count   = 0;
% 
%     for p_index = 1:length(p_path)
%         person_name = p_path{p_index};
%         disp(['>> Processing person: ', person_name]);
% 
%         % "p_now" is the person's folder in the VIDEO path
%         p_now = fullfile(video_root_path, person_name);
%         disp(['>> Checking videos for: ', p_now]);
% 
%         % Gather subfolders for this person
%         v_path_struct = dir(p_now);
%         v_path_names  = sort_nat({v_path_struct.name});
%         v_path_names  = v_path_names(3:end);
% 
%         % Also gather subfolders in the same manner for the "video" side
%         p_now_vedio = fullfile(video_root_path, person_name);
%         v_path_vedio_struct = dir(p_now_vedio);
%         v_path_vedio_names  = sort_nat({v_path_vedio_struct.name});
%         v_path_vedio_names  = v_path_vedio_names(3:end);
% 
%         v_length = min(length(v_path_names), length(v_path_vedio_names));
% 
%         for v_index = 1:v_length
%             v_now = fullfile(p_now, v_path_names{v_index});
%             disp(['>> Found video folder: ', v_now]);
% 
%             % Landmark path for this person (the script only handles "s1")
%             full_s_path = fullfile(Lmk_root_path, person_name);
% 
%             disp(['>> Checking Landmarks in: ', full_s_path]);
%             s_path_struct = dir(full_s_path);
%             s_path = {s_path_struct([s_path_struct.isdir]).name};
%             disp('>> Expected Landmark Folders:');
%             disp(s_path);
%             s_path = s_path(3:end);
% 
%             % Video subfolders
%             v_now_vedio = fullfile(p_now_vedio, v_path_vedio_names{v_index});
%             full_s_path_vedio = fullfile(video_root_path, person_name);
%             disp(['>> Checking Video folder in: ', full_s_path_vedio]);
%             s_path_vedio_struct = dir(full_s_path_vedio);
%             s_path_vedio = {s_path_vedio_struct([s_path_vedio_struct.isdir]).name};
%             disp('>> Expected Video Folders:');
%             disp(s_path_vedio);
%             s_path_vedio = s_path_vedio(3:end);
% 
%             % Match "s1" <--> "source1"
%             falg1 = zeros(1, length(source1));
%             falg2 = zeros(1, length(source2));
%             for source_kind = 1:length(source1)
%                if ismember(source1{source_kind}, s_path)
%                    falg1(source_kind) = 1;
%                end
%                if ismember(source2{source_kind}, s_path_vedio)
%                    falg2(source_kind) = 1;
%                end
%             end
%             s_real = find(falg1 == falg2 & falg1 == 1);
% 
%             disp('>> Checking s_path (Landmarks Subfolders):');
%             disp(s_path);
%             disp('>> Checking s_path_vedio (Video Subfolders):');
%             disp(s_path_vedio);
%             disp(['>> Checking falg1: ', num2str(falg1)]);
%             disp(['>> Checking falg2: ', num2str(falg2)]);
% 
%             if isempty(s_real)
%                 warning('No matching landmarks and videos found for person %s.', person_name);
%                 continue;
%             end
% 
%             for s_index = 1:length(s_real)
%                 % Build final paths
%                 landmark_path = fullfile(Lmk_root_path, person_name, source1{s_real(s_index)}, 'v1', 'source1', 'face_landmarks');
%                 vidio_path    = fullfile(video_root_path, person_name, 'source1', 'video.avi');
%                 HR_path       = fullfile(video_root_path, person_name, 'source1', 'gt_HR.csv');
%                 SpO2_path     = fullfile(video_root_path, person_name, 'source1', 'gt_SpO2.csv');
%                 wave_path     = fullfile(video_root_path, person_name, 'source1', 'wave.csv');
% 
%                 if exist(landmark_path, 'dir') ~= 7
%                     warning('>> ERROR: Landmark directory "%s" does not exist!', landmark_path);
%                     continue;
%                 end
%                 disp(['>> Checking video file: ', vidio_path]);
%                 if exist(vidio_path, 'file') ~= 2
%                     error('>> ERROR: Video file "%s" does not exist!', vidio_path);
%                 end
% 
%                 obj = VideoReader(vidio_path);
%                 numFrames = obj.NumFrames;
%                 numlandmarks = length(dir(landmark_path)) - 2;  % subtract . and ..
% 
%                 disp(['>> Frames: ', num2str(numFrames), ' | Landmarks: ', num2str(numlandmarks)]);
% 
%                 % Only process if frames match landmarks, and #frames>520
%                 if (numFrames == numlandmarks) && (numFrames > 520)
%                     count = count + 1;
%                     disp(['>> count = ', num2str(count)]);
% 
%                     % Read HR, SpO2, wave
%                     HR   = csvread(HR_path,1,0);
%                     SpO2 = csvread(SpO2_path,1,0);
%                     wave = csvread(wave_path,1,0);
% 
%                     % fps
%                     fps = (numFrames / length(HR));
%                     if (fps > 15)
%                         % Create output subfolder
%                         dst_path = fullfile(save_path, person_name, v_path_names{v_index}, source1{s_real(s_index)});
%                         if ~exist(dst_path, 'dir')
%                             mkdir(dst_path);
%                         end
% 
%                         % 1) wave2Pic
%                         wave2Pic(wave, numFrames, dst_path);
% 
%                         % 2) STMap
%                         POS_STMap(vidio_path, landmark_path, dst_path, fps, lmk_num);
% 
%                         % 3) Save HR, SpO2, fps (spline-interpolated)
%                         HR_size   = size(HR);
%                         SpO2_size = size(SpO2);
%                         w_x  = linspace(0,100,HR_size(1));
%                         w_x1 = linspace(0,100,SpO2_size(1));
%                         d_x  = linspace(0,100,numFrames-1);
% 
%                         HR   = spline(w_x, HR, d_x);
%                         SpO2 = spline(w_x1, SpO2, d_x);
% 
%                         fps_path   = fullfile(dst_path, 'fps.mat');
%                         HR_matpath = fullfile(dst_path, 'HR.mat');
%                         SpO2_matpath = fullfile(dst_path, 'SpO2.mat');
% 
%                         save(fps_path, 'fps');
%                         save(HR_matpath, 'HR');
%                         save(SpO2_matpath, 'SpO2');
% 
%                         disp(['>> Finished processing: ', dst_path]);
%                     else
%                         disp('>> Skipped: fps <= 15');
%                     end
%                 else
%                     disp('>> Skipped: #frames != #landmarks or frames <= 520');
%                 end
%             end
%         end
%     end
% 
%     disp(['>> Total valid videos processed: ', num2str(count)]);
% end
function VIPL_Prossessing()
    disp('>> Starting VIPL_Processing with new structure...');
    
    % Paths (adjust to your actual folder layout):
    Lmk_root_path   = 'C:\Users\User\Desktop\Backup Lenovo\D Drive\Testing FYP\NAS-HR\Landmarks';       % Landmarks root
    video_root_path = 'C:\Users\User\Desktop\Backup Lenovo\D Drive\Testing FYP\NAS-HR\video_root_path';    % Videos root
    save_path       = 'C:\Users\User\Desktop\Backup Lenovo\D Drive\Testing FYP\NAS-HR\Output';             % Output folder

    % Get list of person folders (assumed to be the same in both roots)
    person_struct = dir(Lmk_root_path);
    person_names = {person_struct([person_struct.isdir]).name};
    person_names = person_names(~ismember(person_names, {'.','..'}));
    disp(['>> Found persons in dataset: ', strjoin(person_names, ', ')]);
    
    lmk_num = 468;  % Number of landmarks (for MediaPipe)
    count = 0;
    
    for p = 1:length(person_names)
        person_name = person_names{p};
        disp(['>> Processing person: ', person_name]);
        
        % Get version subfolders in landmarks
        lm_version_struct = dir(fullfile(Lmk_root_path, person_name));
        lm_versions = {lm_version_struct([lm_version_struct.isdir]).name};
        lm_versions = lm_versions(~ismember(lm_versions, {'.','..'}));
        lm_versions = sort_nat(lm_versions);
        
        % Get version subfolders in videos
        vid_version_struct = dir(fullfile(video_root_path, person_name));
        vid_versions = {vid_version_struct([vid_version_struct.isdir]).name};
        vid_versions = vid_versions(~ismember(vid_versions, {'.','..'}));
        vid_versions = sort_nat(vid_versions);
        
        % Process only the minimum number of versions present in both roots
        num_versions = min(length(lm_versions), length(vid_versions));
        
        for v = 1:num_versions
            version_name = lm_versions{v};  % (Assuming the ordering matches)
            disp(['>> Processing version: ', version_name]);
            
            % List available source folders in the current version (landmarks)
            lm_source_struct = dir(fullfile(Lmk_root_path, person_name, version_name));
            lm_source_names = {lm_source_struct([lm_source_struct.isdir]).name};
            lm_source_names = lm_source_names(~ismember(lm_source_names, {'.','..'}));
            lm_source_names = sort_nat(lm_source_names);
            
            % List available source folders in the current version (videos)
            vid_source_struct = dir(fullfile(video_root_path, person_name, version_name));
            vid_source_names = {vid_source_struct([vid_source_struct.isdir]).name};
            vid_source_names = vid_source_names(~ismember(vid_source_names, {'.','..'}));
            vid_source_names = sort_nat(vid_source_names);
            
            % Debug: Display the source folder names found
            disp(['   Landmark sources found: ', strjoin(lm_source_names, ', ')]);
            disp(['   Video sources found: ', strjoin(vid_source_names, ', ')]);
            
            % Process each matching source folder (using the minimum count)
            num_sources = min(length(lm_source_names), length(vid_source_names));
            for s = 1:num_sources
                current_lm_src = lm_source_names{s};
                current_vid_src = vid_source_names{s};
                
                % Build full paths for the current source in the current version
                % Landmarks: assume the folder "face_landmarks" is inside the source folder.
                landmark_path = fullfile(Lmk_root_path, person_name, version_name, current_lm_src, 'face_landmarks');
                % Videos: assume the video.avi and CSV files are inside the video source folder.
                vidio_path    = fullfile(video_root_path, person_name, version_name, current_vid_src, 'video.avi');
                HR_path       = fullfile(video_root_path, person_name, version_name, current_vid_src, 'gt_HR.csv');
                SpO2_path     = fullfile(video_root_path, person_name, version_name, current_vid_src, 'gt_SpO2.csv');
                wave_path     = fullfile(video_root_path, person_name, version_name, current_vid_src, 'wave.csv');
                
                % Check if the landmark folder exists
                if exist(landmark_path, 'dir') ~= 7
                    warning('>> ERROR: Landmark directory "%s" does not exist!', landmark_path);
                    continue;
                end
                
                disp(['>> Checking video file: ', vidio_path]);
                if exist(vidio_path, 'file') ~= 2
                    warning('>> ERROR: Video file "%s" does not exist!', vidio_path);
                    continue;
                end
                
                % Read video and count frames; count .dat files in landmark folder
                obj = VideoReader(vidio_path);
                numFrames = obj.NumFrames;
                numlandmarks = length(dir(landmark_path)) - 2;  % subtract '.' and '..'
                disp(['>> Frames: ', num2str(numFrames), ' | Landmark files: ', num2str(numlandmarks)]);
                
                % Only process if the number of frames matches the number of .dat files and if frames > 520
                if (numFrames == numlandmarks) && (numFrames > 520)
                    count = count + 1;
                    
                    % Read CSV files for HR, SpO2, and wave signal
                    HR   = csvread(HR_path, 1, 0);
                    SpO2 = csvread(SpO2_path, 1, 0);
                    wave = csvread(wave_path, 1, 0);
                    
                    % Calculate fps (frames per HR sample)
                    fps = (numFrames / length(HR));
                    if fps > 15
                        % Create output folder organized by person, version, and the current source
                        dst_path = fullfile(save_path, person_name, version_name, current_lm_src);
                        if ~exist(dst_path, 'dir')
                            mkdir(dst_path);
                        end
                        
                        % Process wave: save as a picture
                        wave2Pic(wave, numFrames, dst_path);
                        % Generate spatiotemporal map (STMap) and apply POS
                        POS_STMap(vidio_path, landmark_path, dst_path, fps, lmk_num);
                        
                        % Spline-interpolate HR and SpO2 signals
                        HR_size   = size(HR);
                        SpO2_size = size(SpO2);
                        w_x  = linspace(0,100,HR_size(1));
                        w_x1 = linspace(0,100,SpO2_size(1));
                        d_x  = linspace(0,100,numFrames-1);
                        
                        HR   = spline(w_x, HR, d_x);
                        SpO2 = spline(w_x1, SpO2, d_x);
                        
                        % Save results
                        fps_path    = fullfile(dst_path, 'fps.mat');
                        HR_matpath  = fullfile(dst_path, 'HR.mat');
                        SpO2_matpath = fullfile(dst_path, 'SpO2.mat');
                        
                        save(fps_path, 'fps');
                        save(HR_matpath, 'HR');
                        save(SpO2_matpath, 'SpO2');
                        
                        disp(['>> Finished processing: ', dst_path]);
                    else
                        disp('>> Skipped: fps <= 15');
                    end
                else
                    disp('>> Skipped: #frames != #landmark files or frames <= 520');
                end
            end
        end
    end
    
    disp(['>> Total valid videos processed: ', num2str(count)]);
end
