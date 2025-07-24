""" Operations """  # Module docstring: Defines candidate operations for the search and augmentation phases
import torch  # Import PyTorch
import torch.nn as nn  # Import neural network modules
import genotypes as gt  # Import genotype utilities (to access list of primitives, etc.)

# Dictionary mapping operation names to lambda functions that instantiate the corresponding operation.
# Each operation takes channel count (C), stride, and a boolean flag (affine) as input.
OPS = {
    'none': lambda C, stride, affine: Zero(stride),  # Zero op: outputs zero tensor
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),  # 3x3 average pooling followed by batch norm
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),  # 3x3 max pooling followed by batch norm
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),  # Identity (skip) connection; if stride != 1, reduce dimensions
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),  # 3x3 separable convolution (applied twice inside SepConv)
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),  # 5x5 separable convolution with padding 2
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),  # 7x7 separable convolution with padding 3
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),  # 3x3 dilated convolution with dilation factor 2 and padding 2
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),  # 5x5 dilated convolution with dilation factor 2 and padding 4
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine)  # Factorized convolution using a 7x1 followed by a 1x7 conv
}

# Function to apply DropPath (stochastic depth) to a tensor.
def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:  # Only apply during training and if drop_prob is greater than 0
        keep_prob = 1. - drop_prob  # Probability of keeping the path
        # Create a binary mask with the same batch size and singleton dimensions for spatial dimensions.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)  # Scale x to maintain expectation and then multiply by the mask (in-place)
    return x  # Return the modified tensor

# Module version of DropPath; implemented as an in-place layer.
class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ Initialize DropPath.
        Args:
            p: Probability of dropping a path (set to zero).
        """
        super().__init__()  # Initialize parent class
        self.p = p  # Store the drop probability

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)  # Provide additional string representation

    def forward(self, x):
        drop_path_(x, self.p, self.training)  # Apply drop path on x (in-place)
        return x  # Return the modified x

# Pooling operation with Batch Normalization (applies either max pooling or average pooling)
class PoolBN(nn.Module):
    """
    AvgPool or MaxPool followed by Batch Normalization.
    """
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: String 'max' or 'avg' to select pooling type.
            C: Number of channels for the BatchNorm.
            kernel_size, stride, padding: Parameters for the pooling layer.
            affine: Whether the BatchNorm has learnable affine parameters.
        """
        super().__init__()  # Initialize parent class
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)  # Use max pooling
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)  # Use average pooling (exclude padding in average)
        else:
            raise ValueError()  # Raise error if pool_type is unrecognized
        self.bn = nn.BatchNorm2d(C, affine=affine)  # Initialize BatchNorm with specified channels

    def forward(self, x):
        out = self.pool(x)  # Apply pooling to x
        out = self.bn(out)  # Apply BatchNorm to pooled output
        return out  # Return the result

# Standard Convolution block: Conv2d followed by BatchNorm and ReLU activation.
class StdConv(nn.Module):
    """ Standard conv: ReLU -> Conv -> BN """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()  # Initialize parent class
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),  # Convolution layer
            nn.BatchNorm2d(C_out, affine=affine),  # Batch normalization
            nn.ReLU(),  # ReLU activation
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the sequence

# Factorized Convolution: splits a convolution into two separate convolutions (vertical and horizontal)
class FacConv(nn.Module):
    """ Factorized conv: ReLU -> Conv(Kx1) -> Conv(1xK) -> BN """
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()  # Initialize parent class
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),  # Vertical convolution
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),  # Horizontal convolution
            nn.BatchNorm2d(C_out, affine=affine),  # Batch normalization
            nn.ReLU(),  # ReLU activation
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the factorized convolution block

# Dilated Convolution: Depthwise convolution with dilation, followed by pointwise convolution.
class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv:
        ReLU -> (Dilated depthwise conv) -> Pointwise conv -> BN
        Example: 3x3 conv with dilation=2 has an effective receptive field similar to 5x5.
    """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()  # Initialize parent class
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in, bias=False),
            # Depthwise convolution: each input channel is convolved separately.
            nn.BatchNorm2d(C_out),  # Batch normalization applied after depthwise conv (note: this BN uses C_out channels)
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),  # Pointwise convolution: combine channels
            nn.BatchNorm2d(C_out, affine=affine),  # Batch normalization on pointwise output
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the dilated conv block

# Separable Convolution: Applies depthwise separable convolution twice (using DilConv with dilation=1)
class SepConv(nn.Module):
    """ Depthwise separable conv: applies DilConv (with dilation=1) twice """
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()  # Initialize parent class
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),  # First separable conv
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)  # Second separable conv (no stride)
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the sequential separable conv layers

# Identity operation: returns the input as is.
class Identity(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize parent class

    def forward(self, x):
        return x  # Return the input unchanged

# Zero operation: returns a tensor of zeros with the same shape as input (possibly subsampled if stride > 1).
class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()  # Initialize parent class
        self.stride = stride  # Store the stride value

    def forward(self, x):
        if self.stride == 1:
            return x * 0.  # Multiply input by 0 if no subsampling
        # For stride > 1, slice the tensor with the given stride and multiply by 0
        return x[:, :, ::self.stride, ::self.stride] * 0.

# Factorized Reduce: Reduces spatial resolution and adjusts the number of channels.
class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise convolutions with stride=2.
    """
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()  # Initialize parent class
        self.relu = nn.ReLU()  # ReLU activation before reducing
        # Two parallel 1x1 convolutions that reduce the spatial dimensions (stride=2) and produce half of C_out each.
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)  # Batch normalization on the concatenated output

    def forward(self, x):
        # Apply both convolutions and concatenate their outputs along the channel dimension.
        out = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        out = self.bn(out)  # Apply BatchNorm to the concatenated tensor
        return out  # Return the reduced feature map

# Mixed Operation: Used during architecture search. It computes a weighted sum over all candidate operations.
class MixedOp(nn.Module):
    """ Mixed operation that combines multiple candidate ops """
    def __init__(self, C, stride):
        super().__init__()  # Initialize parent class
        self._ops = nn.ModuleList()  # Create an empty ModuleList to hold each candidate op
        for primitive in gt.PRIMITIVES:  # Iterate over all primitive operation names defined in genotypes.py
            op = OPS[primitive](C, stride, affine=False)  # Instantiate the operation using the OPS dictionary
            self._ops.append(op)  # Append the operation to the list

    def forward(self, x, weights):
        """
        Args:
            x: input tensor
            weights: a list or tensor of weights (probabilities) for each candidate op
        Returns:
            The weighted sum of outputs from each candidate operation.
        """
        # For each op in the list, multiply its output by the corresponding weight and sum them all up.
        return sum(w * op(x) for w, op in zip(weights, self._ops))
