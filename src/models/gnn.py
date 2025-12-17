"""
Structural Encoder: Graph Neural Network (GNN) for Protein Structure

Implements Graph Attention Network (GATv2) to process 3D protein topology
and extract structural embeddings.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool  # type: ignore
    from torch_geometric.data import Data  # type: ignore
except Exception as exc:  # pragma: no cover
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
        # Standard amino acid encoding
        AA_CODES = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20, 'Z': 21
        }
        
        # Physicochemical properties (hydrophobicity, charge, size)
        HYDROPHOBICITY = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5,
            'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
            'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9,
            'Y': -1.3, 'V': 4.2, 'X': 0.0, 'Z': -3.5
        }
        
        features = []
        for i, aa in enumerate(sequence):
            # One-hot encoding (22-dim)
            onehot = torch.zeros(22)
            if aa in AA_CODES:
                onehot[AA_CODES[aa]] = 1.0
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


def main():
    """Test ProteinGNN module."""
    logger.info("Testing ProteinGNN module...")
    
    # Create dummy protein graph
    num_residues = 100
    feature_dim = 24  # 22 (one-hot + AA) + 1 (hydro) + 1 (pLDDT)
    
    # Random features and coordinates
    x = torch.randn(num_residues, feature_dim)
    ca_coords = torch.randn(num_residues, 3) * 10  # Angstroms
    
    # Construct graph
    data = GraphConstruction.construct_ca_graph(ca_coords, x, distance_cutoff=10.0)
    
    logger.info(f"Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Create batch (simulate multiple proteins)
    batch = torch.zeros(num_residues, dtype=torch.long)
    data.batch = batch
    
    # Initialize model
    model = ProteinGNN(input_dim=feature_dim, hidden_dim=64, output_dim=128)
    
    # Forward pass
    with torch.no_grad():
        embedding = model(data)
    
    logger.info(f"Output embedding shape: {embedding.shape}")
    assert embedding.shape == (1, 128), "Unexpected output shape"
    
    logger.info("ProteinGNN test passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
