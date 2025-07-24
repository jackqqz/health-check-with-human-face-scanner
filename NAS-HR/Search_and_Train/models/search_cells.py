""" CNN cell for architecture search """
import torch
import torch.nn as nn
from models import ops


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    # This class defines a search cell for NAS. Each edge in the cell is a mixture of operations, 
    # and the architecture is relaxed to allow continuous optimization.

    def __init__(self, n_nodes, C_pp, C_p, C, reduction_p, reduction):
        """
        Args:
            n_nodes: # of intermediate n_nodes
            C_pp: C_out[k-2]
            C_p : C_out[k-1]
            C   : C_in[k] (current)
            reduction_p: flag for whether the previous cell is reduction cell or not
            reduction: flag for whether the current cell is reduction cell or not
        """
        super().__init__()
        self.reduction = reduction  # Store whether the current cell is a reduction cell.
        self.n_nodes = n_nodes      # Store the number of intermediate nodes in the cell.

        # Preprocessing for the output of cell[k-2].
        if reduction_p:
            # If the previous cell is a reduction cell, the output size of cell[k-2] does not match
            # the input size of the current cell. Use FactorizedReduce to downsample.
            self.preproc0 = ops.FactorizedReduce(C_pp, C, affine=False)
        else:
            # If the previous cell is not a reduction cell, use a standard convolution to match dimensions.
            self.preproc0 = ops.StdConv(C_pp, C, 1, 1, 0, affine=False)
        
        # Preprocessing for the output of cell[k-1].
        self.preproc1 = ops.StdConv(C_p, C, 1, 1, 0, affine=False)

        # Generate the directed acyclic graph (DAG) for the cell.
        self.dag = nn.ModuleList()              # A list of lists to store operations for each edge in the DAG.
        for i in range(self.n_nodes):
            # For each intermediate node, create a list of operations for its incoming edges.
            self.dag.append(nn.ModuleList())
            for j in range(2+i):   # Each node has 2 input nodes plus the previous intermediate nodes.
                # reduction should be used only for input node
                # If the current cell is a reduction cell, apply stride=2 for the first two input nodes.
                stride = 2 if reduction and j < 2 else 1
                # Create a mixed operation (a combination of multiple operations) for the edge.
                op = ops.MixedOp(C, stride)
                self.dag[i].append(op)  # Add the operation to the DAG.


    def forward(self, s0, s1, w_dag):
        """
        Forward pass for the search cell.

        Args:
            s0: Output from cell[k-2].
            s1: Output from cell[k-1].
            w_dag: Weights for the edges in the DAG (used for architecture search).
        """
        # Preprocess the outputs of cell[k-2] and cell[k-1].
        s0 = self.preproc0(s0)  # Apply preprocessing to s0 (output of cell[k-2]).
        s1 = self.preproc1(s1)  # Apply preprocessing to s1 (output of cell[k-1]).

        states = [s0, s1]  # Initialize the list of states with the preprocessed inputs.

        # Iterate over the intermediate nodes and their corresponding weights.
        for edges, w_list in zip(self.dag, w_dag):
            # For each node, compute its output as the weighted sum of its incoming edges.
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)  # Append the current node's output to the list of states.

        # Concatenate the outputs of all intermediate nodes along the channel dimension.
        s_out = torch.cat(states[2:], dim=1)
        return s_out  # Return the final output of the cell.
