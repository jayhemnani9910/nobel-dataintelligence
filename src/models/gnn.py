"""
Structural Encoder: Graph Neural Network (GNN) for Protein Structure

Implements Graph Attention Network (GATv2) to process 3D protein topology
and extract structural embeddings.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import AA_TO_IDX, HYDROPHOBICITY

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torch_geometric is required for `src.models.gnn`. Install it (and its compiled dependencies) "
        "to use graph neural network features."
    ) from exc

logger = logging.getLogger(__name__)


class ProteinGNN(nn.Module):
    """
    Graph Attention Network for protein structure encoding.
    
    Architecture:
    - Input: Protein graph with C-alpha nodes
    - Features: Amino acid identity, physicochemical properties, pLDDT scores
    - Edges: Based on spatial proximity (distance < 10 Angstroms)
    - Output: 128-dim latent embedding
    """
    
    def __init__(self, input_dim: int = 22, hidden_dim: int = 64, 
                 output_dim: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize Protein GNN.
        
        Args:
            input_dim: Input feature dimension (amino acids: 20 + 2 special)
            hidden_dim: Hidden dimension in GAT layers
            output_dim: Output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            # First layer takes hidden_dim, subsequent layers take hidden_dim * num_heads (concat output)
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gat_layers.append(
                GATv2Conv(in_channels, hidden_dim, heads=num_heads,
                         dropout=dropout, add_self_loops=True, 
                         share_weights=False, concat=True)
            )
        
        # Output projection: after GAT layers, output is hidden_dim * num_heads
        self.output_proj = nn.Linear(hidden_dim * num_heads * 2, output_dim)  # *2 for concat of mean+max pooling
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Initialized ProteinGNN with {num_layers} GAT layers")
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            data: PyG Data object with x, edge_index, batch
            
        Returns:
            Graph embeddings: Shape (batch_size, output_dim)
        """
        x = data.x  # (num_nodes, input_dim)
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Apply GAT layers with residual connections
        for i, gat_layer in enumerate(self.gat_layers):
            x_prev = x
            x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = self.dropout(x)
            
            # Residual connection (if dimensions match)
            if x_prev.size(-1) == x.size(-1):
                x = x + x_prev
        
        # Readout layer: combine global mean and max pooling
        if batch is not None:
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
        else:
            # Single graph
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_max = torch.max(x, dim=0, keepdim=True)[0]
        
        # Concatenate pooled features
        x_global = torch.cat([x_mean, x_max], dim=-1)
        
        # Final projection
        embedding = self.output_proj(x_global)
        
        return embedding


class GraphConstruction:
    """Utilities for constructing protein graphs from atomic coordinates."""
    
    @staticmethod
    def construct_ca_graph(ca_coords: torch.Tensor,
                          ca_features: torch.Tensor,
                          distance_cutoff: float = 10.0,
                          edge_features: bool = True) -> Data:
        """
        Construct protein graph from C-alpha coordinates and features.
        
        Args:
            ca_coords: C-alpha coordinates (num_residues, 3)
            ca_features: Node features (num_residues, feature_dim)
            distance_cutoff: Distance threshold for edge creation (Angstroms)
            edge_features: Whether to include distance as edge features
            
        Returns:
            PyG Data object
        """
        num_residues = ca_coords.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(ca_coords, ca_coords)  # (N, N)
        
        # Create edges within cutoff
        edge_index = torch.where(distances < distance_cutoff)
        edge_index = torch.stack(edge_index)
        
        # Remove self-loops
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
        
        # Create edge features (normalized distances)
        if edge_features:
            edge_attr = distances[edge_index[0], edge_index[1]].unsqueeze(1)
            # Normalize to [0, 1]
            edge_attr = 1.0 - (edge_attr / distance_cutoff)
            edge_attr = torch.clamp(edge_attr, 0, 1)
        else:
            edge_attr = None
        
        # Create PyG Data object
        data = Data(
            x=ca_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=ca_coords
        )
        
        return data
    
    @staticmethod
    def construct_residue_features(sequence: str, pldt_scores: torch.Tensor = None) -> torch.Tensor:
        """
        Construct node feature vectors from amino acid sequence.
        
        Features:
        - One-hot amino acid encoding (20 standard AA)
        - Physicochemical properties
        - Optional: pLDDT confidence scores
        
        Args:
            sequence: Amino acid sequence
            pldt_scores: Optional pLDDT scores (num_residues,)
            
        Returns:
            Feature tensor (num_residues, feature_dim)
        """
        features = []
        for i, aa in enumerate(sequence):
            # One-hot encoding (22-dim)
            onehot = torch.zeros(22)
            if aa in AA_TO_IDX:
                onehot[AA_TO_IDX[aa]] = 1.0
            else:
                onehot[20] = 1.0  # Unknown
            
            # Physicochemical features
            hydro = torch.tensor([HYDROPHOBICITY.get(aa, 0.0)])
            
            # pLDDT if available
            if pldt_scores is not None:
                confidence = pldt_scores[i:i+1]
            else:
                confidence = torch.tensor([0.5])  # Default
            
            # Combine features
            residue_features = torch.cat([onehot, hydro, confidence])
            features.append(residue_features)
        
        return torch.stack(features)


