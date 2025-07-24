""" CNN cell for network augmentation """  # Module docstring: Defines the discrete cell used in the augmentation phase
import torch            # Import PyTorch
import torch.nn as nn   # Import neural network module from PyTorch
from models import ops  # Import the operations module that contains building blocks (e.g., Conv, Pooling)
import genotypes as gt  # Import genotype utilities to process the discrete architecture representation

class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete as defined by the genotype.
    """
    def __init__(self, genotype, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            genotype: Discrete architecture description (genotype) defining the operations and connectivity
            C_pp: Number of channels from cell k-2 (two cells before)
            C_p: Number of channels from cell k-1 (previous cell)
            C: Number of channels for the current cell
            reduction_p: Boolean indicating if the previous cell was a reduction cell
            reduction: Boolean indicating if the current cell is a reduction cell
        """
        super().__init__()                      # Initialize the parent class (nn.Module)
        self.reduction = reduction              # Save whether this cell is a reduction cell
        self.n_nodes = len(genotype.normal)     # Number of intermediate nodes is determined by the length of genotype.normal

        # Preprocess the first input:
        if reduction_p:
            # If the previous cell was a reduction cell, adjust the spatial dimensions and channels accordingly
            self.preproc0 = ops.FactorizedReduce(C_pp, C)
        else:
            # Otherwise, use a standard convolution (1x1 conv)
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0)
        # Preprocess the second input always with a standard 1x1 conv
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0)

        # Generate the cell’s directed acyclic graph (DAG) using the genotype
        if reduction:
            gene = genotype.reduce  # For reduction cells, use the reduction gene
            self.concat = genotype.reduce_concat  # And use the corresponding concatenation rule
        else:
            gene = genotype.normal  # For normal cells, use the normal gene
            self.concat = genotype.normal_concat  # And its concatenation rule

        # Use the helper function from genotypes.py to convert the gene into a list of discrete operations (the DAG)
        self.dag = gt.to_dag(C, gene, reduction)

    def forward(self, s0, s1):
        # Preprocess the two inputs using the defined preprocessing layers
        s0 = self.preproc0(s0)  # Preprocess input from cell k-2
        s1 = self.preproc1(s1)  # Preprocess input from cell k-1

        states = [s0, s1]  # Initialize the list of node outputs with the two input nodes
        # Iterate over each set of edges (one for each intermediate node)
        for edges in self.dag:
            # For the current node, compute its output as the sum of outputs from each incoming edge
            # Each edge applies its corresponding operation on one of the previous states (using op.s_idx to index)
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)  # Append the output of the current node to the states list

        # Concatenate the outputs from selected intermediate nodes (as defined by self.concat) along the channel dimension
        s_out = torch.cat([states[i] for i in self.concat], dim=1)
        return s_out  # Return the cell’s final output
