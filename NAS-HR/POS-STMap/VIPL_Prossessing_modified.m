function VIPL_Prossessing_modified()
    disp('>> Starting VIPL_Prossessing_modified with recursive folder search...');
    
    %% Define root paths
    % Adjust these paths to your actual folder structure:
    Lmk_root_path   = 'C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\Landmarks';
    video_root_path = 'C:\Users\User\Documents\Monash\FYP\VIPL-HR Dataset';
    save_path       = 'C:\Users\User\Documents\Monash\FYP\VIPL_Processing_Output';
    
    %% Recursively get all subdirectories under Lmk_root_path
    % We assume that person folders have names that match pattern: '^p\d+$'
    all_dirs = strsplit(genpath(Lmk_root_path), pathsep);
    person_dirs = {};
    for i = 1:length(all_dirs)
        [~, folderName, ~] = fileparts(all_dirs{i});
        if ~isempty(folderName)
            % Use a regular expression to check if folderName starts with 'p' followed by digits.
            if ~isempty(regexp(folderName, '^p\d+$', 'once'))
                person_dirs{end+1} = all_dirs{i}; %#ok<AGROW>
            end
        end
    end
    
    % Sort the person directories naturally
    person_dirs = sort_nat(person_dirs);
    
    disp(['>> Found person folders: ', strjoin(person_dirs, ',\n')]);
    
    lmk_num = 468;  % for MediaPipe landmarks
    count = 0;
    
    %% Loop over each person directory found
    for p = 1:length(person_dirs)
        person_folder = person_dirs{p};
        [~, person_name, ~] = fileparts(person_folder);
        disp(['>> Processing person: ', person_name]);
        
        % In each person folder, assume there are version folders (e.g., v1, v2, ...)
        v_struct = dir(person_folder);
        version_names = {v_struct([v_struct.isdir]).name};
        version_names = version_names(~ismember(version_names, {'.','..'}));
        version_names = sort_nat(version_names);
        
        % For video, we assume a similar structure exists.
        % Determine the relative path of the person folder with respect to Lmk_root_path:
        relative_person_path = strrep(person_folder, Lmk_root_path, '');
        video_person_folder = fullfile(video_root_path, relative_person_path);
        
        disp(['Lmk_root_path: ', Lmk_root_path]);
        disp(['person_folder: ', person_folder]);
        disp(['relative_person_path: ', relative_person_path]);
        disp(['video_person_folder: ', video_person_folder]);

        if ~exist(video_person_folder, 'dir')
            warning('>> No corresponding video folder for person %s', person_name);
            continue;
        end
        
        % List version folders in the video person folder
        vv_struct = dir(video_person_folder);
        video_version_names = {vv_struct([vv_struct.isdir]).name};
        video_version_names = video_version_names(~ismember(video_version_names, {'.','..'}));
        video_version_names = sort_nat(video_version_names);
        
        % Process only the minimum number of versions found in both
        num_versions = min(length(version_names), length(video_version_names));
        for v = 1:num_versions
            version_name = version_names{v};
            disp(['>> Processing version: ', version_name]);
            
            % Get source folders in the current version for landmarks
            lm_src_struct = dir(fullfile(person_folder, version_name));
            lm_source_names = {lm_src_struct([lm_src_struct.isdir]).name};
            lm_source_names = lm_source_names(~ismember(lm_source_names, {'.','..'}));
            lm_source_names = sort_nat(lm_source_names);
            
            % Get source folders in the corresponding video version folder
            vid_src_struct = dir(fullfile(video_person_folder, version_name));
            video_source_names = {vid_src_struct([vid_src_struct.isdir]).name};
            video_source_names = video_source_names(~ismember(video_source_names, {'.','..'}));
            video_source_names = sort_nat(video_source_names);
            
            % Process matching source folders (use the minimum count)
            num_sources = min(length(lm_source_names), length(video_source_names));
            for s = 1:num_sources
                current_lm_src = lm_source_names{s};
                current_vid_src = video_source_names{s};
                
                % Build full paths
                landmark_path = fullfile(person_folder, version_name, current_lm_src, 'face_landmarks');
                vidio_path    = fullfile(video_person_folder, version_name, current_vid_src, 'video.avi');
                HR_path       = fullfile(video_person_folder, version_name, current_vid_src, 'gt_HR.csv');
                SpO2_path     = fullfile(video_person_folder, version_name, current_vid_src, 'gt_SpO2.csv');
                wave_path     = fullfile(video_person_folder, version_name, current_vid_src, 'wave.csv');
                
                if exist(landmark_path, 'dir') ~= 7
                    warning('>> ERROR: Landmark directory "%s" does not exist!', landmark_path);
                    continue;
                end
                disp(['>> Checking video file: ', vidio_path]);
                if exist(vidio_path, 'file') ~= 2
                    warning('>> ERROR: Video file "%s" does not exist!', vidio_path);
                    continue;
                end
                
                % Read video and count frames; count landmark files (.dat) in landmark_path
                obj = VideoReader(vidio_path);
                numFrames = obj.NumFrames;
                numlandmarks = length(dir(landmark_path)) - 2;
                disp(['>> Frames: ', num2str(numFrames), ' | Landmark files: ', num2str(numlandmarks)]);
                
                % Only process if the numbers match and frames > 520
                if (numFrames == numlandmarks) && (numFrames > 520)
                    count = count + 1;
                    % Read CSV files
                    HR   = csvread(HR_path,1,0);
                    SpO2 = csvread(SpO2_path,1,0);
                    wave = csvread(wave_path,1,0);
                    fps = (numFrames / length(HR));
                    if fps > 15
                        % Create output folder: organized by person, version, and source
                        dst_path = fullfile(save_path, person_name, version_name, current_lm_src);
                        if ~exist(dst_path, 'dir')
                            mkdir(dst_path);
                        end
                        % Process wave signal and generate picture
                        wave2Pic(wave, numFrames, dst_path);
                        % Process STMap using POS_STMap function
                        POS_STMap(vidio_path, landmark_path, dst_path, fps, lmk_num);
                        
                        % Spline-interpolate HR and SpO2 signals
                        HR_size = size(HR);
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