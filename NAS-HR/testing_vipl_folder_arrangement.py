import os
import shutil

def pad_subject(subject_str):
    """
    Convert a subject folder name like 'p1' to 'Subject_001'
    by extracting digits and zero-padding to 3 digits.
    """
    # remove any leading non-digits (e.g., 'p')
    num_str = ''.join(filter(str.isdigit, subject_str))
    if not num_str:
        return None
    return f"Subject_{int(num_str):03d}"

def process_vipl(input_root, output_root):
    # Ensure output root exists
    os.makedirs(output_root, exist_ok=True)
    
    # List all subject folders in input_root (e.g., p1, p2, etc.)
    for subject_folder in os.listdir(input_root):
        subject_path = os.path.join(input_root, subject_folder)
        if not os.path.isdir(subject_path):
            continue  # skip if not a directory
        
        subject_name = pad_subject(subject_folder)
        if subject_name is None:
            # If the folder name doesn't have digits, skip
            continue
        
        print(f"Processing {subject_folder} as {subject_name} ...")
        
        # Look for version folders like "v1", "v2", "v3", ...
        # ignoring any top-level "sourceX" folders
        for version_folder in os.listdir(subject_path):
            version_path = os.path.join(subject_path, version_folder)
            if not os.path.isdir(version_path):
                continue
            
            # Only proceed if the folder starts with 'v' or 'V'
            # e.g. v1, v2, v9, etc.
            if not version_folder.lower().startswith('v'):
                # skip top-level "source1" or any other folder
                continue
            
            # Extract version number (remove non-digits)
            ver_num = ''.join(filter(str.isdigit, version_folder))
            if not ver_num:
                # if there's no digit, skip
                continue
            
            # Inside this version folder, look for "source1", "source2", etc.
            for source_folder in os.listdir(version_path):
                source_path = os.path.join(version_path, source_folder)
                if not os.path.isdir(source_path):
                    continue
                
                # e.g., "source1" => parse out '1'
                if not source_folder.lower().startswith('source'):
                    continue
                src_num = ''.join(filter(str.isdigit, source_folder))
                if not src_num:
                    continue
                
                # We expect "img_mvg_full.png" and "HR.mat" inside this folder
                img_file = os.path.join(source_path, "img_mvavg_full.png")
                hr_file  = os.path.join(source_path, "HR.mat")
                
                if not (os.path.isfile(img_file) and os.path.isfile(hr_file)):
                    print(f"  Skipping {subject_folder} {version_folder} {source_folder}: missing file(s)")
                    continue
                
                # Construct the output folder name, e.g., "Subject_001_V1S1"
                out_folder_name = f"{subject_name}_V{ver_num}S{src_num}"
                out_folder_path = os.path.join(output_root, out_folder_name)
                
                # Create the output folder and subfolders
                os.makedirs(out_folder_path, exist_ok=True)
                label_folder = os.path.join(out_folder_path, "Label_CSI")
                stmap_folder = os.path.join(out_folder_path, "STMap")
                os.makedirs(label_folder, exist_ok=True)
                os.makedirs(stmap_folder, exist_ok=True)
                
                # Copy HR.mat to Label_CSI
                dest_hr = os.path.join(label_folder, "HR.mat")
                shutil.copy2(hr_file, dest_hr)
                
                # Copy and rename img_mvg_full.png to STMap_YUV_Align_CSI_POS.png in STMap
                dest_img = os.path.join(stmap_folder, "STMap_YUV_Align_CSI_POS.png")
                shutil.copy2(img_file, dest_img)
                
                print(f"  Copied files to {out_folder_name}")

if __name__ == "__main__":
    # Adjust these paths to match your environment:
    input_path = r"C:\Users\User\Documents\Monash\FYP\VIPL_Processing_Output"
    output_path = r"C:\Users\User\Documents\Monash\FYP\VIPL_STMaps_HR_Full"
    
    process_vipl(input_path, output_path)
    print("Processing complete.")
