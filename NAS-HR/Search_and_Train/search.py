""" Search cell """
# This code fork from https://github.com/khanrc/pt.darts
# modified by Hao Lu for Heart Rate Estimation

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import numpy as np
import MyDataset
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_cnn import SearchCNNController
from architect import Architect
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as transF
from torchvision import transforms

# Parse command-line arguments and configuration settings for the search phase.
config = SearchConfig()
device = torch.device("cuda")

# Set up TensorBoard logging
# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb")) # Initialize TensorBoard SummaryWriter with log directory.
writer.add_text('config', config.as_markdown(), 0)              # Log configuration parameters as markdown text.

# Set up logger to output messages both to file and console.
logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))  # Create a logger that writes to a log file.
config.print_params(logger.info)                                                    # Print configuration parameters using the logger.

def main():
    torch.cuda.empty_cache()

    logger.info("Logger is set - training start")

    # Define the root directory where the raw VIPL dataset is stored.
    # fileRoot = r"C:/Users/User/Documents/Monash/FYP/VIPL_STMaps_HR_Full" 
    # fileRoot = r"C:/Users/User/Documents/Monash/FYP/PURE_Full"  # Set the root directory for preprocessed data.
    fileRoot = r"C:/Users/User/Documents/Monash/FYP/pure_gb_4x4_Full"

    # Define the root path for preprocessed STMap images, appending fold number and fold index.
    saveRoot = r"C:/Users/User/Documents/Monash/FYP/pure_gb_4x4_" + str(config.fold_num) + str(config.fold_index)
    
    n_classes = 1        # Set the number of output classes
    input_channels = 3   # Set the number of input chnnels  (RGB)
    
    # set default gpu device id
    torch.cuda.set_device(config.gpus[0])

    # Define image normalization transformation with mean and std values.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Transformation to convert images to PyTorch tensors.
    toTensor = transforms.ToTensor()
    resize = transforms.Resize(size=(64, 300)) # Resize transformation: resize images to 64x300.
    
    # Set random seeds for reproducibility.
    np.random.seed(config.seed)                 # Set NumPy random seed.
    torch.manual_seed(config.seed)              # Set PyTorch CPU seed.
    torch.cuda.manual_seed_all(config.seed)     # Set PyTorch GPU seeds for all GPUs.

    # Enable cuDNN benchmark mode for improved performance.
    torch.backends.cudnn.benchmark = True  # 网络加速

    # If reData flag is set, perform cross-validation split.
    if config.reData == 1:
        test_index, train_index = MyDataset.CrossValidation(fileRoot, fold_num=config.fold_num, fold_index=config.fold_index)
        print("### TEST INDEX ###")
        print(test_index)
        print(len(test_index))
        print("")
        print("### TRAIN INDEX ###")
        print(train_index)
        print(len(train_index))

        # # Generate index files for the training set and save them in the designated folder
        Train_Indexa = MyDataset.getIndex(fileRoot, train_index, saveRoot + '_Train', 'STMap_YUV_Align_CSI_POS.png', 15, 300)
        
        # # Generate index files for the test/validation set similarly
        # Test_Indexa = MyDataset.getIndex(fileRoot, test_index, saveRoot + '_Test', 'STMap_YUV_Align_CSI_POS.png', 15, 300)
    

    # Create training dataset: load preprocessed STMap images with the specified transformations.
    train_data = MyDataset.Data_STMap(root_dir=(saveRoot + '_Train'), frames_num=300,
                                    transform=transforms.Compose([resize, toTensor, normalize]))
    
    print("")
    print("!!!! TRAIN DATA !!!!")
    print(train_data.datalist)
    print(len(train_data))

    net_crit = nn.L1Loss().to(device) # Define the loss function (L1 loss for regression) and move it to GPU.
    
    # Initialize the search network controller with the given input channels, initial channels, number of classes, number of layers, and loss function.
    model = SearchCNNController(input_channels, config.init_channels, n_classes, config.layers,
                                net_crit, device_ids=config.gpus)
    
    # Print the initial genotype
    genotype = model.genotype()
    # print("Initial Genotype:", genotype)

    # Initialize network weights using the defined weight initialization routine.
    model._init_weight()
    model = model.to(device) # Move the model to GPU.

    # Set up the optimizer for network weights using SGD.
    w_optim = torch.optim.SGD(model.weights(), config.w_lr, momentum=config.w_momentum,
                              weight_decay=config.w_weight_decay)
    # The following alternative optimizer (Adam) is commented out.
    # w_optim = torch.optim.Adam(model.weights(), config.w_lr)


    # Set up the optimizer for the architecture parameters (alphas) using Adam.
    alpha_optim = torch.optim.Adam(model.alphas(), config.alpha_lr, betas=(0.5, 0.999),
                                   weight_decay=config.alpha_weight_decay)


    # Split the training data into two parts (train/validation) for training weights and updating architecture parameters.
    n_train = len(train_data)           # Total number of training samples.
    split = n_train // 2                # Split point: first half for training weights, second half for validation (for alpha update).
    indices = list(range(n_train))      # Create a list of all sample indices.
    
    # Create a sampler for the training subset.
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    # Create a sampler for the validation subset.
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    # Create DataLoader for training data using the train_sampler.
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=train_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    # Create DataLoader for validation data using the valid_sampler.
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    
    # Print sampler indices
    print("\nTrain Sampler Indices:", list(train_sampler))
    print("\nValidation Sampler Indices:", list(valid_sampler))

    # Print the number of batches in each DataLoader
    print("\nNumber of Batches in Train Loader:", len(train_loader))
    print("\nNumber of Batches in Validation Loader:", len(valid_loader))

    # Print the shapes of the first batch from each DataLoader
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\nTrain Loader - Batch {batch_idx}: Data Shape: {data.shape}, Target Shape: {target.shape}")
        break

    for batch_idx, (data, target) in enumerate(valid_loader):
        print(f"\nValidation Loader - Batch {batch_idx}: Data Shape: {data.shape}, Target Shape: {target.shape}")
        break

    # Set up a learning rate scheduler using cosine annealing, adjusting the weight optimizer's learning rate over epochs.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min)
    
    # Initialize the Architect instance to manage architecture parameter updates.
    architect = Architect(model, config.w_momentum, config.w_weight_decay)
    
    
    # Begin the training loop over the specified number of epochs.
    best_losses = 100                   # Initialize best loss to a high value.
    
    for epoch in range(config.epochs):  # 40
        lr_scheduler.step()             # Update the learning rate according to the scheduler.
        lr = lr_scheduler.get_lr()[0]   # Get the current learning rate (from the first parameter group).
        model.print_alphas(logger)      # Log the current state of architecture parameters (alphas).
        
        # Call the training function for one epoch.
        train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)
        
        # After training, validate the model on the validation set.
        cur_step = (epoch+1) * len(train_loader)                    # Compute the current step (for logging purposes).
        losses = validate(valid_loader, model, epoch, cur_step)     # Evaluate the model and get the average validation loss.
        
        # Compute the discrete genotype (architecture) from the learned continuous parameters.
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))   # Log the current genotype.

        # Save the model checkpoint if the current validation loss is the best so far.
        if losses < best_losses:
            best_losses = losses        # Update best loss.
            best_genotype = genotype    # Save the current best genotype.
            is_best = True              # Mark this checkpoint as best.
        else:
            is_best = False             # Not the best checkpoint.
        utils.save_checkpoint(model, config.path, is_best)      # Save the checkpoint to disk.
        print("")

    logger.info("Best Genotype = {}".format(best_genotype))     # Log the best genotype discovered during the search phase.


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    losses = utils.AverageMeter()               # Create an AverageMeter to track the average training loss.

    cur_step = epoch*len(train_loader)          # Initialize the current step counter.
    writer.add_scalar('train/lr', lr, cur_step) # Log the current learning rate to TensorBoard.

    model.train()                               # Set the model to training mode.

    # Iterate over the training and validation loaders simultaneously.
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        # Move training inputs and labels to GPU.
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        # Move validation inputs and labels to GPU.
        val_X, val_y = val_X.to(device, non_blocking=True), val_y.to(device, non_blocking=True)
        trn_y = torch.squeeze(trn_y)    # Remove extra dimensions from training labels.
        val_y = torch.squeeze(val_y)    # Remove extra dimensions from validation labels.
        N = trn_X.size(0)               # Get the batch size.
        
        # Phase 2: Update architecture parameters (alpha) using validation loss.
        alpha_optim.zero_grad() # Zero the gradients for the alpha optimizer.
        # Compute the unrolled gradient for alphas (updates based on a virtual step).
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()      # Update architecture parameters.

        # phase 1. child network step (w)
        # Phase 1: Update the network weights using training loss.
        w_optim.zero_grad()                         # Zero the gradients for the weights optimizer.
        logits = model(trn_X)                       # Forward pass: compute the output predictions.
        logits = torch.squeeze(logits)              # Remove extra dimensions from the predictions.
        loss = model.criterion(logits, trn_y)       # Compute the loss (L1 loss between predictions and labels).
        loss.backward()                             # Backpropagate the loss to compute gradients.
        # Perform gradient clipping to avoid exploding gradients.
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()                              # Update the network weights.

        losses.update(loss.item(), N)               # Update the loss meter with the current loss and batch size.

        # Log training progress at defined intervals.
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses))

        writer.add_scalar('train/loss', loss.item(), cur_step) # Log the loss to TensorBoard.
        cur_step += 1   # Increment the current step counter.

# NO USAGE
def train_model(train_loader, valid_loader, model, w_optim, lr, epoch):
    losses = utils.AverageMeter()                   # Initialize an AverageMeter for tracking loss.
    cur_step = epoch*len(train_loader)              # Calculate the starting step for the current epoch
    writer.add_scalar('train/lr', lr, cur_step)     # Log the learning rate
    model.train()    # Set the model to training mode.

    # Iterate over the training and validation sets.
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        trn_y = torch.squeeze(trn_y)
        N = trn_X.size(0)
        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        logits = torch.squeeze(logits)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()
        losses.update(loss.item(), N)
        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Pre_Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses))
        writer.add_scalar('train/loss', loss.item(), cur_step)
        cur_step += 1


def validate(valid_loader, model, epoch, cur_step):
    losses = utils.AverageMeter()               # Create an AverageMeter to track validation loss.
    model.eval()                                # Set the model to evaluation mode.
    with torch.no_grad():                       # Disable gradient calculation for validation.
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)                       # Get batch size.

            logits = model(X)                   # Forward pass to get predictions.
            logits = torch.squeeze(logits)      # Squeeze predictions.
            loss = model.criterion(logits, y)   # Compute validation loss.

            losses.update(loss.item(), N)       # Update average loss.

            # Log validation progress at defined intervals.
            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses))

    writer.add_scalar('val/loss', losses.avg, cur_step) # Log average validation loss to TensorBoard.

    # logger.info("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.epochs, top1.avg))

    return losses.avg   # Return the average validation loss.


if __name__ == "__main__":
    main()
