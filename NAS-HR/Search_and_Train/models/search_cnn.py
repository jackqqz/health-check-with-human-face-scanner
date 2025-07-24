""" CNN for architecture search """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from torch.nn.parallel._functions import Broadcast  # Import Broadcast function for multi-GPU support
import logging

# Helper function to broadcast a list of tensors to multiple GPUs.
def broadcast_list(l, device_ids):
    """ Broadcasting list """
    # Use the Broadcast function to create copies of list elements for each GPU device.
    l_copies = Broadcast.apply(device_ids, *l)
    # Group the copied elements into sublists, one for each device.
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]
    return l_copies  # Return the list of copies grouped by device

# Define the CNN model used during architecture search.
class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()              # Initialize the parent class (nn.Module)
        self.C_in = C_in                # Save the number of input channels
        self.C = C                      # Save the number of starting channels
        self.n_classes = n_classes      # Save the number of output classes
        self.n_layers = n_layers        # Save the number of layers (cells)

        # Build the stem network (initial layers to process input image)
        C_cur = 32                      # Initial number of channels after the stem
        self.stem = nn.Sequential(
            nn.BatchNorm2d(C_in),       # Apply Batch Normalization to the input
            nn.Conv2d(C_in, C_cur, 5, 2, 2, bias=False),  # 5x5 convolution with stride=2 and padding=2
            nn.BatchNorm2d(C_cur),      # BatchNorm for the output of the convolution
            nn.ReLU(),                  # ReLU activation function
            nn.Conv2d(C_cur, C_cur, 3, 2, 1, bias=False),  # 3x3 convolution with stride=2 and padding=1
            nn.BatchNorm2d(C_cur),                          # BatchNorm again
            nn.ReLU(),                  # Another ReLU activation
        )

       # Initialize variables to manage cell connectivity.
        # C_pp: output channels of cell k-2, C_p: output channels of cell k-1, C_cur: current cell channels.
        C_pp, C_p, C_cur = C_cur, C_cur, C_cur
        self.cells = nn.ModuleList()  # Create an empty ModuleList to store the cells
        reduction_p = False  # Flag to indicate if the previous cell was a reduction cell

        # Build the sequence of cells
        for i in range(n_layers):
            # Determine if the current cell should be a reduction cell (downsampling).
            if i in [1*n_layers//6, 3*n_layers//6, 5*n_layers//6]:
                C_cur *= 2  # Double the channel count for reduction cells
                reduction = True  # Mark current cell as a reduction cell
            else:
                reduction = False  # Otherwise, it's a normal cell
            # Create a SearchCell with the given parameters.
            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction  # Update the flag for the next cell based on current reduction status
            self.cells.append(cell)  # Append the cell to the ModuleList
            C_cur_out = C_cur * n_nodes  # Output channels of the cell equals current channels times the number of nodes
            C_pp, C_p = C_p, C_cur_out  # Update channel values for the next iteration

        # Define the global pooling and classification head
        self.gap = nn.Sequential(
            nn.Conv2d(C_p, 512, 3, 2, 1, bias=False),  # Convolution to adjust channels and spatial dimensions
            nn.BatchNorm2d(512),  # Batch normalization
            nn.AdaptiveAvgPool2d(1)  # Adaptive average pooling to produce a 1x1 feature map
        )
        self.linear = nn.Linear(512, n_classes)  # Linear layer to produce final output

    def forward(self, x, weights_normal, weights_reduce):
        # Forward pass of the network.
        s0 = s1 = self.stem(x)  # Pass input through the stem and set both previous states s0 and s1
        # Iterate over each cell in the network
        for cell in self.cells:
            # Choose appropriate architecture weights: use reduction weights if the cell is a reduction cell.
            weights = weights_reduce if cell.reduction else weights_normal
            # Update the states: new state is computed from previous states s0 and s1 using the cell.
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.gap(s1)  # Apply global average pooling to the final state
        out = out.view(out.size(0), -1)  # Flatten the output tensor
        logits = self.linear(out)  # Compute the final logits through the linear layer
        return logits  # Return the logits (predictions)


# Define the controller for the search CNN model that supports multi-GPU training.
class SearchCNNController(nn.Module):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, C_in, C, n_classes, n_layers, criterion, n_nodes=3, stem_multiplier=3,
                 device_ids=None):
        """
        Args:
            C_in: Number of input channels.
            C: Starting number of channels.
            n_classes: Number of output classes.
            n_layers: Number of layers (cells) in the network.
            criterion: Loss function to optimize.
            n_nodes: Number of intermediate nodes in each cell.
            stem_multiplier: Multiplier to determine stem channel count.
            device_ids: List of GPU device IDs for multi-GPU training.
        """
        super().__init__()  # Initialize the parent nn.Module
        self.n_nodes = n_nodes  # Save the number of intermediate nodes
        self.criterion = criterion  # Save the loss function
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))  # Use all available GPUs if none specified
        self.device_ids = device_ids  # Save the device IDs

        # Initialize architecture parameters (alphas) for normal and reduction cells.
        n_ops = len(gt.PRIMITIVES)  # Get the number of primitive operations from the genotype module
        self.alpha_normal = nn.ParameterList()  # ParameterList for normal cell architecture parameters
        self.alpha_reduce = nn.ParameterList()  # ParameterList for reduction cell architecture parameters

        # For each intermediate node, initialize a parameter tensor with shape (i+2, n_ops)
        for i in range(n_nodes):
            self.alpha_normal.append(nn.Parameter(1e-3 * torch.randn(i+2, n_ops)))
            self.alpha_reduce.append(nn.Parameter(1e-3 * torch.randn(i+2, n_ops)))

        # Collect all architecture parameters into a list for convenience.
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:  # Only include parameters whose name contains 'alpha'
                self._alphas.append((n, p))

        # Initialize the actual search network.
        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)

    def forward(self, x):
        # Compute softmax-normalized weights for each edge in the cell for both normal and reduction cells.
        weights_normal = [F.softmax(alpha, dim=-1) for alpha in self.alpha_normal]
        weights_reduce = [F.softmax(alpha, dim=-1) for alpha in self.alpha_reduce]

        if len(self.device_ids) == 1:
            # If only one GPU is used, perform a normal forward pass.
            return self.net(x, weights_normal, weights_reduce)

        # Multi-GPU support:
        xs = nn.parallel.scatter(x, self.device_ids)  # Scatter the input tensor across multiple GPUs
        wnormal_copies = broadcast_list(weights_normal, self.device_ids)  # Broadcast normal cell weights to all GPUs
        wreduce_copies = broadcast_list(weights_reduce, self.device_ids)  # Broadcast reduction cell weights to all GPUs

        replicas = nn.parallel.replicate(self.net, self.device_ids)  # Replicate the network on each GPU
        # Apply the network on each GPU in parallel with corresponding inputs and weights
        outputs = nn.parallel.parallel_apply(replicas,
                                             list(zip(xs, wnormal_copies, wreduce_copies)),
                                             devices=self.device_ids)
        return nn.parallel.gather(outputs, self.device_ids[0])  # Gather the outputs from all GPUs and return them

    def loss(self, X, y):
        logits = self.forward(X)  # Forward pass: compute model outputs (logits)
        return self.criterion(torch.squeeze(logits), torch.squeeze(y))  # Compute loss using the criterion and return it

    def print_alphas(self, logger):
        # Temporarily remove any formatter from the logger to print raw alpha values.
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, dim=-1))  # Log softmax-normalized alpha values for normal cells

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, dim=-1))  # Log softmax-normalized alpha values for reduction cells
        logger.info("#####################")

        # Restore the original formatters for logger handlers.
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)

    def _init_weight(self):
        # Initialize weights for all submodules in the model.
        for m in self.modules():   # Iterate through all modules in the network.
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)  # Use Kaiming Normal initialization for Conv2d layers.
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Use Xavier Normal initialization for Linear layers.

    def genotype(self):
        # Convert the continuous architecture parameters (alphas) into a discrete genotype.
        gene_normal = gt.parse(self.alpha_normal, k=2)  # Parse the alpha parameters for normal cells, selecting top-2 edges.
        gene_reduce = gt.parse(self.alpha_reduce, k=2)  # Parse the alpha parameters for reduction cells similarly.
        concat = range(2, 2+self.n_nodes)  # Define the concatenation indices: usually all intermediate nodes.

        # Return the genotype as a namedtuple with fields for normal and reduction cells and their concat lists.
        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()  # Return an iterator over the network's weight parameters.

    def named_weights(self):
        return self.net.named_parameters()  # Return an iterator over (name, parameter) pairs of network weights.

    def alphas(self):
        for n, p in self._alphas:
            yield p  # Yield each architecture parameter

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p  # Yield each (name, parameter) pair for architecture parameters