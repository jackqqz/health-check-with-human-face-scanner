""" Training augmented model """
# This code fork from https://github.com/khanrc/pt.darts
# modified by Hao Lu for Heart Rate Estimation

import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import MyDataset
import torchvision.transforms.functional as transF
from torchvision import transforms
import utils
from models.augment_cnn import AugmentCNN
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from thop import profile
from thop import clever_format

# Create a configuration object by parsing command-line arguments
config = AugmentConfig()

device = torch.device("cuda")

# Initialize TensorBoard writer to log training information (logs will be saved under config.path/tb)
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)  # Log the configuration as markdown text

# Set up a logger to output messages both to console and to a log file
logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
# Print all configuration parameters using the logger
config.print_params(logger.info)


def main():
    logger.info("Logger is set - training start")

    # Define paths for raw data and preprocessed STMaps (for cross-validation)
    # fileRoot = r"C:/Users/User/Documents/Monash/FYP/VIPL_STMaps_HR_Full"
    fileRoot = r"C:/Users/User/Documents/Monash/FYP/pure_gb_4x4_Full"        
    
    # Construct the save root path for preprocessed data (including fold numbers and indices)
    # saveRoot = r"C:/Users/User/Documents/Monash/FYP/VIPL" + str(config.fold_num) + str(config.fold_index)
    saveRoot = r"C:/Users/User/Documents/Monash/FYP/pure_gb_4x4_" + str(config.fold_num) + str(config.fold_index)

    n_classes = 1                               # The regression task outputs a single value (HR)
    input_channels = 3                          # Input images have 3 channels (RGB)
    input_size = np.array([64, 300])            # Expected size of the input STMap (height=64, width=300)
    
    # Define image transformations:
    # Normalize using ImageNet statistics
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    toTensor = transforms.ToTensor()            # Convert images to PyTorch tensors
    resize = transforms.Resize(size=(64, 300))  # Resize images to 64x300
    
    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])
    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    
    # Enable cuDNN benchmark mode to optimize performance for fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    # get data with meta info
    # If reData flag is set, perform cross-validation splitting and generate index files
    if config.reData == 1:
        
        # Split the raw dataset into test and training indices based on the fold settings
        # test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=config.fold_num,
        #                                                     fold_index=config.fold_index)
        # print("\nTest Index: ", test_index)
        # print("\nTrain Index: ", train_index)

        train_index, val_index, test_index = MyDataset.SplitDataset(fileRoot, train_ratio=0.7, val_ratio=0.2)
        print("\nTrain Index: ", train_index)
        print("\nValidation Index: ", val_index)
        print("\nTest Index: ", test_index)

        # Generate index files for the training set and save them in the designated folder
        Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'STMap_YUV_Align_CSI_POS.png', 15, 300)
        
        # Generate index files for the validation set similarly
        Val_Indexa = MyDataset.getIndex(fileRoot, val_index, saveRoot + '_Val', 'STMap_YUV_Align_CSI_POS.png', 15, 300)

        # Generate index files for the test set
        Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test', 'STMap_YUV_Align_CSI_POS.png', 15, 300)
    
    # Create training and validation datasets using the generated index files
    train_data = MyDataset.Data_STMap(root_dir=(saveRoot + '_Train'), frames_num=300,
                                      transform=transforms.Compose([resize, toTensor, normalize]))
    valid_data = MyDataset.Data_STMap(root_dir=(saveRoot + '_Val'), frames_num=300,
                                      transform=transforms.Compose([resize, toTensor, normalize]))
    
     # Create DataLoaders for training and validation datasets
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,                    # Shuffle training data
                                               num_workers=config.workers,      # Number of worker threads for loading data
                                               pin_memory=True)                 # Pin memory for faster GPU transfers
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,                   # Do not shuffle validation data
                                               num_workers=config.workers,
                                               pin_memory=True)
    
    # Define the loss function (L1 loss for regression) and move it to GPU
    criterion = nn.L1Loss().to(device)

    # Construct the model name using configuration parameters
    Model_name = config.name + 'fn' + str(config.fold_num) + 'fi' + str(config.fold_index)
    
    # Check if auxiliary loss should be used (if aux_weight > 0)
    use_aux = config.aux_weight > 0.
    
    # If reTrain flag is set, load a pre-trained model from disk
    if config.reTrain == 1:
        model_path = os.path.join(config.path, Model_name + 'best.pth.tar')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pre-trained model file not found: {model_path}")
        
        model = torch.load(os.path.join(config.path, Model_name + 'best.pth.tar'), map_location=device)
        print('Loaded pre-trained model:', Model_name)

        # Wrap the model for multi-GPU training if necessary
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)
    else:
        # Otherwise, instantiate a new AugmentCNN model using the provided input size, channels, layers, etc.
        model = AugmentCNN(input_size, input_channels, config.init_channels, n_classes, config.layers,
                           use_aux, config.genotype)
        model._init_weight()    # Initialize the model's weights
        model = nn.DataParallel(model, device_ids=config.gpus).to(device)   # Enable multi-GPU training

    # Compute the model size in MB and log it
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    
    # Set up an optimizer (Adam) for updating the model's weights with the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), config.lr)
    best_losses = 10    # Initialize the best loss (for tracking improvement)

    # Training loop: iterate over the number of epochs specified in the configuration
    for epoch in range(config.epochs):
        # Train the model for one epoch
        train(train_loader, model, optimizer, criterion, epoch)
        # After training, evaluate the model on the validation set
        cur_step = (epoch+1) * len(train_loader)
        best_losses = validate(valid_loader, model, criterion, epoch, cur_step, best_losses)
    # Log the final best loss achieved during training
    logger.info("Final best Losses@1 = {:.4%}".format(best_losses))


def train(train_loader, model, optimizer, criterion, epoch):
    losses = utils.AverageMeter()                        # Initialize an object to keep track of the average training loss
    cur_step = epoch * len(train_loader)                 # Calculate the starting step number for the current epoch
    cur_lr = optimizer.param_groups[0]['lr']             # Get the current learning rate from the optimizer
    logger.info("Epoch {} LR {}".format(epoch + 1, cur_lr))  # Log the current epoch and learning rate
    writer.add_scalar('train/lr', cur_lr, cur_step)      # Log the learning rate to TensorBoard
    
    model.train()  # Set the model to training mode
    
    # Iterate over the training DataLoader
    for step, (X, y) in enumerate(train_loader):
        # Move the batch data (input and target) to GPU
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        N = X.size(0)                   # Get the batch size
        optimizer.zero_grad()           # Reset gradients in the optimizer
        logits = model(X)               # Forward pass: compute predictions
        loss = criterion(logits, y)     # Compute the loss (L1 loss)
        
        # If using auxiliary loss, add it here (currently commented out)
        # if config.aux_weight > 0.:
        #     loss += config.aux_weight * criterion(aux_logits, y)
        
        loss.backward()  # Backpropagate the loss to compute gradients
        # Clip gradients to avoid explosion; using a threshold from the configuration
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()                # Update model weights using the optimizer
        losses.update(loss.item(), N)   # Update the average loss meter with the current loss and batch size
        
        # Log training progress at specified intervals (or at the end of the epoch)
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses))
        writer.add_scalar('train/loss', loss.item(), cur_step)  # Log the batch loss to TensorBoard
        cur_step += 1                                           # Increment the step counter


def validate(valid_loader, model, criterion, epoch, cur_step, best_losses):
    losses = utils.AverageMeter()       # Initialize an AverageMeter to track validation loss
    model.eval()                        # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    HR_pr_temp = []                     # List to store predicted heart rates from the model
    HR_rel_temp = []                    # List to store the corresponding ground-truth heart rates
    
    # Disable gradient calculations for validation
    with torch.no_grad():
        # Iterate over the validation DataLoader
        for step, (X, y) in enumerate(valid_loader):
            # Move validation data to GPU
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)                   # Batch size
            logits = model(X)               # Forward pass: compute predictions
            loss = criterion(logits, y)     # Compute the loss
            losses.update(loss.item(), N)   # Update the validation loss meter
            
            # Append predictions and ground-truth values to temporary lists for later evaluation
            HR_pr_temp.extend(logits.data.cpu().numpy())
            HR_rel_temp.extend(y.data.cpu().numpy())
            
            # Log validation progress at specified intervals
            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses))
        
        # Evaluate performance using a custom evaluation function (e.g., compute MAE, RMSE, etc.)
        me, std, mae, rmse, mer, p = utils.MyEval(HR_pr_temp, HR_rel_temp)
        logger.info(
            "Epoch {} Evaluation Metrics: | me: {:.4f} | std: {:.4f} | mae: {:.4f} | rmse: {:.4f} | mer: {:.4f} | p: {:.4f}".format(
                epoch + 1, me, std, mae, rmse, mer, p
            )
        )
        
        # Save the predicted and ground-truth HR values to MATLAB files for further analysis
        io.savemat(os.path.join(config.path, str(config.fold_index) + 'HR_pr.mat'), {'HR_pr': HR_pr_temp})
        io.savemat(os.path.join(config.path, str(config.fold_index) + 'HR_rel.mat'), {'HR_rel': HR_rel_temp})
        
        # Save the model checkpoint if the current validation loss is better than the best seen so far
        if best_losses > losses.avg:
            best_losses = losses.avg
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, config.path, is_best)  # Save the model checkpoint

    # Log the average validation loss to TensorBoard
    writer.add_scalar('val/loss', losses.avg, cur_step)

    return best_losses  # Return the updated best loss value


# If this script is executed as the main program, call the main() function to start training
if __name__ == "__main__":
    main()