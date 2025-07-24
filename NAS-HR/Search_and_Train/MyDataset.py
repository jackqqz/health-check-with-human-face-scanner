# -*- coding: UTF-8 -*-
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import csv
import scipy.io as scio
import torchvision.transforms.functional as transF
import torchvision.transforms as transforms
from PIL import Image

# Define a simple transformation function for an image.
def transform(image):
    image = transF.resize(image, size=(300, 600))   # Resize the image to 300x600 pixels
    image = transF.to_tensor(image)                 # Convert the resized image to a PyTorch tensor.
    # Normalize the tensor with given mean and standard deviation.
    image = transF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

# Define a custom dataset class for STMap data.
class Data_STMap(Dataset):
    def __init__(self, root_dir, frames_num, transform = None):
        self.root_dir = root_dir                # Save the root directory where data files are located.
        self.frames_num = int(frames_num)       # Save the number of frames to consider per sample.
        self.datalist = os.listdir(root_dir)    # List all files in the root directory.
        # Filter to include only .mat files in the directory
        self.datalist = [f for f in os.listdir(root_dir) if f.endswith('.mat')]
        self.num = len(self.datalist)           # Store the total number of samples.
        self.transform = transform              # Save the transformation to be applied to the images.
        if not self.check_integrity():          # Check if the dataset directory exists and is valid.
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

    def __len__(self):
        return self.num     # Return the number of samples in the dataset.

    def __getitem__(self, idx):
        idx = idx                                                       # (Redundant assignment; idx remains unchanged.)
        img_name = 'STMap'                                              # Define the folder name containing the STMap images.
        STMap_name = 'STMap_YUV_Align_CSI_POS.png'                      # Define the filename for the STMap image.
        nowPath = os.path.join(self.root_dir, self.datalist[idx])       # Construct full path to the current sample's MATLAB file.
        temp = scio.loadmat(nowPath)
        nowPath = str(temp['Path'][0])                                  # Convert MATLAB char array to Python string
        Step_Index = int(temp['Step_Index'])                            # Convert MATLAB double to Python int
        STMap_Path = os.path.join(nowPath, img_name)                    # Construct the path to the folder containing the STMap image.

        gt_name = os.path.join('Label_CSI', 'HR.mat')                                   # Define the relative path to the ground-truth HR MATLAB file.
        gt_path = os.path.join(nowPath, gt_name)                        # Construct the full path to the HR file.

        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        gt = scio.loadmat(gt_path)['HR']                                # Load the HR data from the MATLAB file.
        gt = np.array(gt.astype('float32')).reshape(-1)                 # Convert HR data to a float32 NumPy array and flatten it.
        gt = np.nanmean(gt[Step_Index:Step_Index + self.frames_num])    # Compute the mean HR over a sequence of frames (ignoring NaNs).
        gt = gt.astype('float32')                                       # Ensure the HR value is of type float32.

        # 读取图片序列
        feature_map = cv2.imread(os.path.join(STMap_Path, STMap_name))  # Read the STMap image using OpenCV.
        feature_map = feature_map[:, Step_Index:Step_Index + self.frames_num, :]    # Crop the image to use frames from Step_Index to Step_Index + frames_num.
        for c in range(feature_map.shape[2]):       # Loop over each color channel (e.g., R, G, B)
            for r in range(feature_map.shape[0]):    # Loop over each row in the image.
                # Normalize each row for the current channel to the range 0-255.
                feature_map[r, :, c] = 255 * ((feature_map[r, :, c] - np.min(feature_map[r, :, c])) / (0.00001 +
                            np.max(feature_map[r, :, c]) - np.min(feature_map[r, :, c])))
        # Convert the NumPy array to a PIL Image.
        feature_map = Image.fromarray(feature_map)
        
        # If a transformation is provided,
        if self.transform:
            feature_map = self.transform(feature_map)   # apply it to the feature map.
        # 归一化 (Chinese: "Normalization")
        return (feature_map, gt)    # Return a tuple of the processed feature map and the corresponding HR label.

    def check_integrity(self): 
        # Check if the root directory exists.
        if not os.path.exists(self.root_dir): 
            return False
        else:
            return True
        
# Function to create cross-validation splits.
def CrossValidation(root_dir, fold_num=5,fold_index=0):
    datalist = os.listdir(root_dir)         # List all files (or subdirectories) in the root directory.
    # datalist.sort(key=lambda x: int(x))   # Optionally, sort the list numerically
    num = len(datalist)                     # Get the total number of samples.
    test_num = round(((num/fold_num) - 2))  # Calculate the number of test samples per fold (subtracting 2 for possible non-data files).
    train_num = num - test_num              # Calculate the number of training samples (not used further here).
    test_index = datalist[fold_index*test_num:fold_index*test_num + test_num-1]                 # Determine test sample indices for the current fold.
    train_index = datalist[0:fold_index*test_num] + datalist[fold_index*test_num + test_num:]   # Determine training sample indices (all samples not in test_index).
    return test_index, train_index          # Return the test and training indices.

def SplitDataset(root_dir, train_ratio=0.7, val_ratio=0.2):
    datalist = os.listdir(root_dir)  # List all files in the root directory
    datalist.sort()  # Optionally, sort the list to ensure consistent splits
    num = len(datalist)  # Total number of samples

    # Calculate the number of samples for each split
    train_num = int(num * train_ratio)
    val_num = int(num * val_ratio)
    test_num = num - train_num - val_num  # Remaining samples go to the test set

    # Split the data
    train_index = datalist[:train_num]  # First 70% for training
    val_index = datalist[train_num:train_num + val_num]  # Next 20% for validation
    test_index = datalist[train_num + val_num:]  # Last 10% for testing

    return train_index, val_index, test_index

# Function to generate index files for sub-sequences in the dataset.
def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    Index_path = []                                 # Initialize an empty list to store index file paths.
    if not os.path.exists(save_path):               # Check if the save directory exists.
        os.makedirs(save_path)                      # Create the save directory if it does not exist.
    for sub_file in filesList:                      # Loop through each sample in the provided file list.
        now = os.path.join(root_path, sub_file)     # Construct the full path to the current sample.
        img_path = os.path.join(now, os.path.join('STMap', Pic_path))   # Construct the path to the specific STMap image file.
        print(f"Trying to read image: {img_path}")
        if not os.path.exists(img_path):
            print(f"Image file does not exist: {img_path}")
        temp = cv2.imread(img_path)                 # Read the STMap image.
        if temp is None:
            print(f"Failed to read image: {img_path}")
        Num = temp.shape[1]                         # Get the width (number of columns) of the image.
        Res = Num - frames_num - 1  # 可能是Diff数据 # Calculate the remaining width after accounting for the desired number of frames (may account for differences).
        Step_num = int(Res/Step)                    # Determine the number of steps (sub-sequences) that can be generated by dividing the remaining width by the step size.
        for i in range(Step_num):                   # Loop over each possible step.
            Step_Index = i*Step                     # Calculate the starting index for the current step
            temp_path = sub_file + '_' + str(1000 + i) + '_.mat'    # Generate a filename for the index file, including the sample name and a unique identifier.
            scio.savemat(os.path.join(save_path, temp_path), {'Path': now, 'Step_Index': Step_Index})    # Save the sample path and Step_Index in a MATLAB file.
            Index_path.append(temp_path)            # Add the filename to the list of index paths.
    return Index_path                               # Return the list of generated index file paths.

