""" CNN for network augmentation """  # Module docstring: Defines the final CNN used for training (augment phase)
import torch  # Import PyTorch library
import torch.nn as nn  # Import neural network modules from PyTorch
from models.augment_cells import AugmentCell  # Import the cell definition used for building the network
from models import ops  # Import the operations (primitive building blocks) used in the cells

class AugmentCNN(nn.Module):
    """ Augmented CNN model """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 stem_multiplier=3):
        """
        Args:
            input_size: Tuple or array specifying the height and width of the input (assumes square if not)
            C_in: Number of input channels (e.g., 3 for RGB images)
            C: Number of starting channels for the network
            n_classes: Number of output classes (for HR regression, typically 1)
            n_layers: Number of cells (layers) in the network
            auxiliary: Boolean indicating whether to use auxiliary classifiers/losses
            genotype: The discrete architecture description (obtained from search.py)
            stem_multiplier: Multiplier for the number of channels in the stem block
        """
        super().__init__()              # Initialize the parent class (nn.Module)
        self.C_in = C_in                # Store the number of input channels
        self.C = C                      # Store the starting number of channels
        self.n_classes = n_classes      # Store the number of output classes
        self.n_layers = n_layers        # Store the number of cells/layers
        self.genotype = genotype        # Save the genotype which defines the cell architecture

        C_cur = 32  # Set initial channel number after the stem
        
        # Define the stem network: initial layers that process the input image (STMap)
        self.stem = nn.Sequential(
            nn.BatchNorm2d(C_in),                           # Batch normalization on the input
            nn.Conv2d(C_in, C_cur, 5, 2, 2, bias=False),    # 5x5 convolution with stride 2, padding 2
            nn.BatchNorm2d(C_cur),                          # Batch normalization after convolution
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(C_cur, C_cur, 3, 2, 1, bias=False),   # 3x3 convolution with stride 2, padding 1
            nn.BatchNorm2d(C_cur),                          # Batch normalization
            nn.ReLU(),  # ReLU activation
        )

        # Initialize variables to hold output channel sizes for previous cells:
        C_pp, C_p, C_cur = C_cur, C_cur, C_cur  # C_pp: cell k-2, C_p: cell k-1, C_cur: current cell's base channels

        self.cells = nn.ModuleList()    # Create an empty ModuleList to store the cells of the network
        reduction_p = False             # Flag to indicate if the previous cell was a reduction cell
        for i in range(n_layers):  # Iterate over the number of cells/layers
            # Determine if the current cell should be a reduction cell (for downsampling)
            if i in [1*n_layers//6, 3*n_layers//6, 5*n_layers//6]:
                C_cur *= 2  # Double the number of channels in a reduction cell
                reduction = True  # Mark this cell as a reduction cell
            else:
                reduction = False  # Otherwise, it is a normal cell
            # Instantiate an AugmentCell with the given genotype and channel sizes
            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction  # Update flag: current cell's type becomes previous for next iteration
            self.cells.append(cell)  # Append the cell to the ModuleList
            C_cur_out = C_cur * len(cell.concat)  # Compute output channels of this cell: base channels * number of concatenated nodes
            C_pp, C_p = C_p, C_cur_out  # Update channel sizes for the next cell

        # Define the global average pooling and final linear classifier
        self.gap = nn.Sequential(
            nn.Conv2d(C_p, 512, 3, 2, 1, bias=False),   # Convolution to adjust channels and spatial dimensions
            nn.BatchNorm2d(512),                        # Batch normalization
            nn.AdaptiveAvgPool2d(1)                     # Adaptive average pooling to produce a 1x1 output per channel
        )
        self.linear = nn.Linear(512, n_classes)  # Final linear layer to produce the output (e.g., HR)

    def _init_weight(self):
        # Initialize weights for convolutional and linear layers throughout the model
        for m in self.modules():  # Iterate over all submodules
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)  # Kaiming (He) initialization for Conv2d layers
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)        # Xavier initialization for Linear layers

    def forward(self, x):
        s0 = s1 = self.stem(x)  # Pass input through the stem, assign output to both s0 and s1 (for two previous cells)
        # Pass through each cell sequentially:
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)       # Update the cell inputs: s0 becomes previous, s1 becomes current cell output
        out = self.gap(s1)                  # Apply global average pooling to the final cell output
        out = out.view(out.size(0), -1)     # Flatten the pooled features into a vector
        logits = self.linear(out)           # Pass flattened vector through the final linear layer
        return torch.squeeze(logits)        # Squeeze the output (remove unnecessary dimensions)

    def drop_path_prob(self, p):
        """ Set drop path probability for all DropPath modules in the network """
        for module in self.modules():  # Iterate through all modules in the model
            if isinstance(module, ops.DropPath_):  # If the module is a DropPath_ operation
                module.p = p  # Set its drop probability to p
