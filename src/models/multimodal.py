"""
Multimodal Architecture: Vibro-Structural Fusion

Combines structural (GNN) and spectral (CNN) embeddings into a unified
architecture for stability and function prediction.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VibroStructuralFusion(nn.Module):
    """
    Fusion layer combining structural and spectral embeddings.
    
    Implements multiple fusion strategies:
    - Concatenation (baseline)
    - Bilinear fusion (element-wise interaction)
    - Cross-attention (learnable interaction weights)
    """
    
    def __init__(self, latent_dim: int = 128, fusion_type: str = "bilinear",
                 dropout: float = 0.1):
        """
        Initialize fusion layer.
        
        Args:
            latent_dim: Dimension of input embeddings
            fusion_type: 'concat', 'bilinear', or 'attention'
            dropout: Dropout probability
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.output_dim = 2 * latent_dim
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(latent_dim, latent_dim, latent_dim)
            self.output_dim = latent_dim
        elif fusion_type == "attention":
            # Cross-attention: compute attention weights
            self.q_proj = nn.Linear(latent_dim, latent_dim)
            self.k_proj = nn.Linear(latent_dim, latent_dim)
            self.v_proj = nn.Linear(latent_dim, latent_dim)
            self.scale = 1.0 / (latent_dim ** 0.5)
            self.output_dim = latent_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, structural_emb: torch.Tensor, 
                spectral_emb: torch.Tensor) -> torch.Tensor:
        """
        Fuse structural and spectral embeddings.
        
        Args:
            structural_emb: (batch_size, latent_dim)
            spectral_emb: (batch_size, latent_dim)
            
        Returns:
            Fused embedding: (batch_size, output_dim)
        """
        if self.fusion_type == "concat":
            fused = torch.cat([structural_emb, spectral_emb], dim=-1)
        
        elif self.fusion_type == "bilinear":
            fused = self.bilinear(structural_emb, spectral_emb)
            fused = F.relu(fused)
        
        elif self.fusion_type == "attention":
            # Cross-attention mechanism
            q = self.q_proj(structural_emb)  # (B, latent_dim)
            k = self.k_proj(spectral_emb)    # (B, latent_dim)
            v = self.v_proj(spectral_emb)    # (B, latent_dim)
            
            # Scalar gating between modalities (0..1)
            gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)  # (B, 1)
            
            # Weighted sum of value with structural embedding
            fused = gate * v + (1 - gate) * structural_emb  # (B, latent_dim)
        
        return self.dropout(fused)


class VibroStructuralModel(nn.Module):
    """
    Complete multimodal architecture for protein analysis.
    
    Architecture:
    ├─ Structural Branch (GNN)
    │  └─ Output: 128-dim structural embedding
    ├─ Spectral Branch (1D CNN)
    │  └─ Output: 128-dim spectral embedding
    ├─ Fusion Layer
    │  └─ Output: Combined embedding
    └─ Task-Specific Heads
       ├─ Novozymes (Regression): Tm prediction
       └─ CAFA 5 (Classification): GO term prediction
    """
    
    def __init__(self, latent_dim: int = 128, gnn_input_dim: int = 24,
                 fusion_type: str = "bilinear", dropout: float = 0.1,
                 num_go_terms: int = 10000):
        """
        Initialize Vibro-Structural model.
        
        Args:
            latent_dim: Dimension of latent embeddings
            gnn_input_dim: Input dimension for GNN
            fusion_type: Fusion strategy ('concat', 'bilinear', 'attention')
            dropout: Dropout probability
            num_go_terms: Number of GO terms for CAFA 5
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.gnn_input_dim = gnn_input_dim
        self.num_go_terms = num_go_terms
        
        # Import models here to avoid circular dependencies.
        # Support both package execution (`python -m src.models.multimodal`)
        # and direct script execution (`python src/models/multimodal.py`).
        try:
            from .gnn import ProteinGNN
            from .cnn import SpectralCNN
        except ImportError:  # pragma: no cover
            from src.models.gnn import ProteinGNN
            from src.models.cnn import SpectralCNN
        
        # Encoders
        self.gnn_encoder = ProteinGNN(input_dim=gnn_input_dim, 
                                     hidden_dim=64,
                                     output_dim=latent_dim)
        self.cnn_encoder = SpectralCNN(input_channels=1,
                                      hidden_channels=32,
                                      output_dim=latent_dim)
        
        # Fusion layer
        self.fusion = VibroStructuralFusion(latent_dim=latent_dim,
                                           fusion_type=fusion_type,
                                           dropout=dropout)
        
        # Global features injection (optional)
        self.global_feature_proj = nn.Sequential(
            nn.Linear(3, 32),  # 3 global features: s_vib, sasa, zpe
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Novozymes stability head (regression)
        novozymes_dim = self.fusion.output_dim + 32  # fused + global
        self.novozymes_head = nn.Sequential(
            nn.Linear(novozymes_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single scalar output (Tm prediction)
        )
        
        # CAFA 5 function head (multi-label classification)
        cafa_dim = self.fusion.output_dim + 32 + 1  # fused + global + taxon_emb
        self.cafa_head = nn.Sequential(
            nn.Linear(cafa_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_go_terms)  # Multi-label logits
        )
        
        # Taxon embeddings (for CAFA 5)
        self.taxon_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=1)
        
        logger.info("Initialized VibroStructuralModel")
    
    def forward(self, graph_data, spectra, global_features=None, 
                taxon_ids=None, task: str = "novozymes"):
        """
        Forward pass through multimodal model.
        
        Args:
            graph_data: PyG Data object (protein structure)
            spectra: Spectral tensor (batch_size, 1, 1000)
            global_features: Optional (batch_size, 3) - [s_vib, sasa, zpe]
            taxon_ids: Optional (batch_size,) - taxon indices for CAFA 5
            task: 'novozymes' or 'cafa5'
            
        Returns:
            Predictions: Regression (float) or logits (multi-label)
        """
        # Encode structure
        struct_emb = self.gnn_encoder(graph_data)  # (B, latent_dim)
        
        # Encode spectra
        spec_emb = self.cnn_encoder(spectra)  # (B, latent_dim)
        
        # Fuse embeddings
        fused_emb = self.fusion(struct_emb, spec_emb)  # (B, fusion_dim)
        
        # Process global features
        if global_features is not None:
            global_proj = self.global_feature_proj(global_features)
        else:
            # Default: zero features
            global_proj = torch.zeros(fused_emb.size(0), 32, 
                                     device=fused_emb.device)
        
        # Concatenate for task head
        combined_features = torch.cat([fused_emb, global_proj], dim=-1)
        
        if task == "novozymes":
            output = self.novozymes_head(combined_features)
            return output  # (B, 1)
        
        elif task == "cafa5":
            # Add taxon embeddings
            if taxon_ids is not None:
                taxon_emb = self.taxon_embedding(taxon_ids)  # (B, 1)
            else:
                # Default: zero taxon embedding
                taxon_emb = torch.zeros(fused_emb.size(0), 1, 
                                       device=fused_emb.device)
            combined_features = torch.cat([combined_features, taxon_emb], dim=-1)
            
            output = self.cafa_head(combined_features)
            return output  # (B, num_go_terms)
        
        else:
            raise ValueError(f"Unknown task: {task}")


class RankingHead(nn.Module):
    """
    Specialized head for pairwise ranking (Novozymes competition).
    
    For ranking mutations, we treat each mutation pair as a training example
    and optimize a ranking loss (e.g., MarginRankingLoss).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1):
        """Initialize ranking head."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single scalar score
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking scores for mutations.
        
        Args:
            embeddings: (batch_size, input_dim)
            
        Returns:
            Scores: (batch_size, 1)
        """
        return self.network(embeddings)


class MultiLabelClassificationHead(nn.Module):
    """
    Multi-label classification head for CAFA 5 (GO term prediction).
    """
    
    def __init__(self, input_dim: int, num_labels: int,
                 hidden_dim: int = 512, dropout: float = 0.1):
        """Initialize classification head."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict GO terms.
        
        Args:
            embeddings: (batch_size, input_dim)
            
        Returns:
            Logits: (batch_size, num_labels)
        """
        return self.network(embeddings)


def main():
    """Test complete multimodal architecture."""
    logger.info("Testing VibroStructuralModel...")
    
    # Create dummy inputs
    batch_size = 8
    
    from torch_geometric.data import Batch
    from src.models.gnn import GraphConstruction
    
    num_residues = 100
    ca_coords = torch.randn(num_residues, 3) * 10
    ca_features = torch.randn(num_residues, 24)
    
    graphs = [
        GraphConstruction.construct_ca_graph(ca_coords.clone(), ca_features.clone())
        for _ in range(batch_size)
    ]
    graph_data = Batch.from_data_list(graphs)
    
    # Dummy spectra
    spectra = torch.randn(batch_size, 1, 1000)
    
    # Global features
    global_features = torch.randn(batch_size, 3)
    
    # Initialize model
    model = VibroStructuralModel(latent_dim=128, num_go_terms=10000)
    
    # Test Novozymes task
    logger.info("Testing Novozymes task...")
    with torch.no_grad():
        output_novozymes = model(graph_data, spectra, global_features, 
                                task="novozymes")
    logger.info(f"Novozymes output shape: {output_novozymes.shape}")
    
    # Test CAFA 5 task
    logger.info("Testing CAFA 5 task...")
    taxon_ids = torch.randint(0, 1000, (batch_size,))
    with torch.no_grad():
        output_cafa = model(graph_data, spectra, global_features, 
                           taxon_ids, task="cafa5")
    logger.info(f"CAFA 5 output shape: {output_cafa.shape}")
    
    logger.info("VibroStructuralModel tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Allow running as a script: `python src/models/multimodal.py`
    import os
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import torch
    main()
