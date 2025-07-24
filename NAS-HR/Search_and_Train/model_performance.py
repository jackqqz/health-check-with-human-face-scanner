import os
import torch
import MyDataset
from torchvision import transforms
import numpy as np
from genotypes import Genotype
from models.augment_cnn import AugmentCNN

# Define paths and parameters
fileRoot = r"C:/Users/User/Documents/Monash/FYP/PURE_Full"
saveRoot = r"C:/Users/User/Documents/Monash/FYP/pure_green_5x5_50"
model_path = os.path.join(r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search_and_Train\augments\pure_green_5x5", "model_best.pth.tar")  # Path to the saved model
log_file_path = os.path.join(r"C:/Users/User/Documents/Monash/FYP/health-check-with-human-face-scanner/NAS-HR/Search_and_Train/augments/pure_green_5x5", "test_results.txt")  # Path to the log file

# Define hyperparameters
batch_size = 32  # Batch size for the DataLoader
num_workers = 0  # Number of worker threads for loading data

# Define image transformations (same as during training)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
toTensor = transforms.ToTensor()
resize = transforms.Resize(size=(64, 300))

# Prepare the test dataset and DataLoader
test_data = MyDataset.Data_STMap(root_dir=(saveRoot + '_Test'), frames_num=300,
                                 transform=transforms.Compose([resize, toTensor, normalize]))
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          num_workers=num_workers,
                                          pin_memory=True)

# Load the saved model
device = torch.device("cuda")

input_size = [64, 300]          # From your code
input_channels = 3              # RGB STMap
init_channels = 36              # Or whatever config.init_channels you used
n_classes = 1
layers = 6                      # Or config.layers
use_aux = False                 # Unless you trained with auxiliary outputs
genotype = Genotype(normal=[[('sep_conv_3x3', 0), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 0), ('skip_connect', 1)], [('avg_pool_3x3', 2), ('max_pool_3x3', 3)]], normal_concat=range(2, 5), reduce=[[('skip_connect', 0), ('avg_pool_3x3', 1)], [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], [('skip_connect', 0), ('sep_conv_5x5', 2)]], reduce_concat=range(2, 5))

model = AugmentCNN(input_size, input_channels, init_channels, n_classes, layers, use_aux, genotype)

# Load checkpoint
checkpoint = torch.load(model_path, map_location=torch.device("cuda:0"))

# Robust loading
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    # Saved as state_dict (normal case)
    state_dict = checkpoint['state_dict']

    # If wrapped by DataParallel ("module." prefix), fix it
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # remove "module." prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
else:
    # Saved full model (like DataParallel or Model)
    model = checkpoint

# Move to GPU and eval mode
model = model.to(device)
model.eval()

# Define the loss function (optional, for computing test loss)
criterion = torch.nn.L1Loss()

model = model.to(device)  # Move the model to the appropriate device

# Evaluate the model on the test set
test_loss = 0
HR_pr_temp = []  # Predicted heart rates
HR_rel_temp = []  # Ground-truth heart rates

with torch.no_grad():  # Disable gradient computation for evaluation
    for X, y in test_loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        # Print the ground truth values
        # print(f"Ground Truth (y): {y.cpu().numpy()}")

        logits = model(X)  # Forward pass

        # Print the model's predictions
        # print(f"Model Predictions (logits): {logits.cpu().numpy()}")
        # break  # Only print for the first batch to avoid excessive output

        loss = criterion(logits, y)  # Compute loss (optional)
        test_loss += loss.item()

        # Store predictions and ground-truth values
        # HR_pr_temp.extend(logits.view(-1).cpu().numpy())
        # HR_rel_temp.extend(y.view(-1).cpu().numpy())
        HR_pr_temp.extend(logits.flatten().cpu().numpy())
        HR_rel_temp.extend(y.flatten().cpu().numpy())


# Compute average test loss
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Evaluate performance (e.g., MAE, RMSE)
HR_pr_temp = np.array(HR_pr_temp)
HR_rel_temp = np.array(HR_rel_temp)
# mae = np.mean(np.abs(HR_pr_temp - HR_rel_temp))  # Mean Absolute Error
# rmse = np.sqrt(np.mean((HR_pr_temp - HR_rel_temp) ** 2))  # Root Mean Squared Error
temp = HR_pr_temp - HR_rel_temp
mae = np.sum(np.abs(temp))/len(temp)
rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))

# Define an accuracy threshold (e.g., within ±5 BPM is considered accurate)
accuracy_threshold = 1
correct_predictions = np.sum(np.abs(HR_pr_temp - HR_rel_temp) <= accuracy_threshold)
accuracy = (correct_predictions / len(HR_rel_temp)) * 100  # Accuracy in percentage

# Log results to a text file
with open(log_file_path, "w") as log_file:
    log_file.write(f"Test Loss: {test_loss:.4f}\n")
    log_file.write(f"Test MAE: {mae:.4f}\n")
    log_file.write(f"Test RMSE: {rmse:.4f}\n")
    log_file.write(f"Test Accuracy: {accuracy:.2f}%\n")

print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")