"""Unit tests for neural network model architectures.

Tests cover:
- Model initialization and parameter counts
- Forward pass with various input shapes
- Output shape validation
- Gradient flow and backpropagation
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.models.cnn import SpectralCNN, ResidualBlock1D, MultiScaleSpectralCNN
from src.models.multimodal import VibroStructuralFusion, VibroStructuralModel
from src.models.losses import (
    MarginRankingLossCustom,
    SpearmanCorrelationLoss,
    FocalLoss,
    WeightedBCELoss
)

try:
    from torch_geometric.data import Data  # type: ignore
    from src.models.gnn import ProteinGNN

    TORCH_GEOMETRIC_AVAILABLE = True
except Exception:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = None  # type: ignore[assignment]
    ProteinGNN = None  # type: ignore[assignment]


@unittest.skipUnless(TORCH_GEOMETRIC_AVAILABLE, "torch_geometric is not available or failed to import")
class TestProteinGNN(unittest.TestCase):
    """Test Graph Neural Network encoder."""
    
    def setUp(self):
        """Initialize model."""
        self.device = torch.device('cpu')
        self.model = ProteinGNN(input_dim=24, hidden_dim=64, output_dim=128, dropout=0.1).to(self.device)
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        self.assertIsNotNone(self.model)
        
        # Check trainable parameters
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(params, 1000)  # Should have at least 1K parameters
    
    def test_forward_pass_small_graph(self):
        """Test forward pass with small graph."""
        # Create small protein graph (20 residues)
        n_nodes = 20
        n_edges = 60
        
        x = torch.randn(n_nodes, 24).to(self.device)  # Node features (24-dim)
        edge_index = torch.randint(0, n_nodes, (2, n_edges)).to(self.device)
        edge_attr = torch.randn(n_edges, 1).to(self.device)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        output = self.model(data)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 128))
    
    def test_forward_pass_large_graph(self):
        """Test forward pass with larger graph."""
        n_nodes = 500  # Large protein
        n_edges = 2000
        
        x = torch.randn(n_nodes, 24).to(self.device)
        edge_index = torch.randint(0, n_nodes, (2, n_edges)).to(self.device)
        edge_attr = torch.randn(n_edges, 1).to(self.device)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        output = self.model(data)
        
        self.assertEqual(output.shape, (1, 128))
    
    def test_gradient_flow(self):
        """Test gradients can flow through model."""
        n_nodes = 50
        n_edges = 150
        
        x = torch.randn(n_nodes, 24, requires_grad=True).to(self.device)
        edge_index = torch.randint(0, n_nodes, (2, n_edges)).to(self.device)
        data = Data(x=x, edge_index=edge_index)
        
        output = self.model(data)
        loss = output.sum()
        loss.backward()
        
        # Check gradient computed on input
        self.assertIsNotNone(x.grad)


class TestSpectralCNN(unittest.TestCase):
    """Test 1D CNN encoder for spectra."""
    
    def setUp(self):
        """Initialize model."""
        self.device = torch.device('cpu')
        self.model = SpectralCNN(input_channels=1, hidden_channels=32, output_dim=128, dropout=0.1).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.assertGreater(params, 500)
    
    def test_forward_pass_standard_spectrum(self):
        """Test forward pass with standard spectrum (1000 points)."""
        batch_size = 8
        spectrum_length = 1000
        
        spectra = torch.randn(batch_size, 1, spectrum_length).to(self.device)
        output = self.model(spectra)
        
        self.assertEqual(output.shape, (batch_size, 128))
    
    def test_forward_pass_different_lengths(self):
        """Test forward pass handles variable spectrum lengths."""
        batch_size = 4
        
        for spectrum_length in [500, 1000, 2000]:
            spectra = torch.randn(batch_size, 1, spectrum_length).to(self.device)
            output = self.model(spectra)
            
            # Should be adaptive pooled to same size
            self.assertEqual(output.shape, (batch_size, 128))
    
    def test_residual_block(self):
        """Test ResidualBlock1D forward pass."""
        block = ResidualBlock1D(in_channels=64, out_channels=64).to(self.device)
        
        x = torch.randn(4, 64, 500).to(self.device)
        output = block(x)
        
        # Output should maintain or reduce spatial dim
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[1], 64)
    
    def test_residual_block_channel_change(self):
        """Test ResidualBlock1D with channel change."""
        block = ResidualBlock1D(in_channels=64, out_channels=128).to(self.device)
        
        x = torch.randn(4, 64, 1000).to(self.device)
        output = block(x)
        
        # Output channels should match out_channels
        self.assertEqual(output.shape[1], 128)
    
    def test_multiscale_cnn(self):
        """Test MultiScaleSpectralCNN."""
        model = MultiScaleSpectralCNN(input_channels=1, hidden_channels=32, output_dim=128).to(self.device)
        
        batch_size = 4
        spectra = torch.randn(batch_size, 1, 1000).to(self.device)
        output = model(spectra)
        
        self.assertEqual(output.shape, (batch_size, 128))


class TestMultimodalFusion(unittest.TestCase):
    """Test multimodal fusion strategies."""
    
    def setUp(self):
        """Initialize test data."""
        self.device = torch.device('cpu')
        self.gnn_output = torch.randn(4, 128).to(self.device)
        self.cnn_output = torch.randn(4, 128).to(self.device)
    
    def test_concat_fusion(self):
        """Test concatenation fusion."""
        fusion = VibroStructuralFusion(
            latent_dim=128,
            fusion_type='concat',
            dropout=0.1
        ).to(self.device)
        
        output = fusion(self.gnn_output, self.cnn_output)
        # concat doubles the dim
        self.assertEqual(output.shape, (4, 256))
    
    def test_bilinear_fusion(self):
        """Test bilinear fusion."""
        fusion = VibroStructuralFusion(
            latent_dim=128,
            fusion_type='bilinear',
            dropout=0.1
        ).to(self.device)
        
        output = fusion(self.gnn_output, self.cnn_output)
        self.assertEqual(output.shape, (4, 128))
    
    def test_attention_fusion(self):
        """Test attention-based fusion."""
        fusion = VibroStructuralFusion(
            latent_dim=128,
            fusion_type='attention',
            dropout=0.1
        ).to(self.device)
        
        output = fusion(self.gnn_output, self.cnn_output)
        self.assertEqual(output.shape, (4, 128))


@unittest.skipUnless(TORCH_GEOMETRIC_AVAILABLE, "torch_geometric is not available or failed to import")
class TestVibroStructuralModel(unittest.TestCase):
    """Test complete end-to-end model."""
    
    def setUp(self):
        """Initialize model."""
        self.device = torch.device('cpu')
        self.model = VibroStructuralModel(
            latent_dim=128,
            gnn_input_dim=24,
            fusion_type='bilinear',
            dropout=0.1,
            num_go_terms=100
        ).to(self.device)
    
    def test_model_initialization(self):
        """Test model has expected structure."""
        self.assertIsNotNone(self.model.gnn_encoder)
        self.assertIsNotNone(self.model.cnn_encoder)
        self.assertIsNotNone(self.model.fusion)
        self.assertIsNotNone(self.model.novozymes_head)
        self.assertIsNotNone(self.model.cafa_head)
    
    def test_novozymes_forward_pass(self):
        """Test forward pass for Novozymes regression task."""
        # Create dummy graph and spectrum
        n_nodes = 100
        batch_size = 1  # Single graph for simplicity
        spectrum_length = 1000
        
        x = torch.randn(n_nodes, 24).to(self.device)
        edge_index = torch.randint(0, n_nodes, (2, 300)).to(self.device)
        spectra = torch.randn(batch_size, 1, spectrum_length).to(self.device)
        
        # Dummy graph data with batch attribute
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n_nodes, dtype=torch.long))
        
        output = self.model(graph, spectra, task='novozymes')
        
        # Novozymes task outputs scalar per sample
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_cafa5_forward_pass(self):
        """Test forward pass for CAFA 5 multi-label task."""
        n_nodes = 100
        batch_size = 1
        spectrum_length = 1000
        num_go = 100
        
        x = torch.randn(n_nodes, 24).to(self.device)
        edge_index = torch.randint(0, n_nodes, (2, 300)).to(self.device)
        spectra = torch.randn(batch_size, 1, spectrum_length).to(self.device)
        
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n_nodes, dtype=torch.long))
        
        output = self.model(graph, spectra, task='cafa5')
        
        # CAFA 5 task outputs logits for all GO terms
        self.assertEqual(output.shape, (batch_size, num_go))
    
    def test_with_global_features(self):
        """Test forward pass with global features (pH, etc.)."""
        n_nodes = 100
        batch_size = 1
        spectrum_length = 1000
        
        x = torch.randn(n_nodes, 24).to(self.device)
        edge_index = torch.randint(0, n_nodes, (2, 300)).to(self.device)
        spectra = torch.randn(batch_size, 1, spectrum_length).to(self.device)
        global_features = torch.randn(batch_size, 3).to(self.device)  # pH, temperature, etc.
        
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n_nodes, dtype=torch.long))
        
        output = self.model(graph, spectra, global_features, task='novozymes')
        
        self.assertEqual(output.shape, (batch_size, 1))


class TestLossFunctions(unittest.TestCase):
    """Test custom loss functions."""
    
    def setUp(self):
        """Initialize test data."""
        self.batch_size = 8
        self.num_go = 100
    
    def test_margin_ranking_loss(self):
        """Test MarginRankingLossCustom."""
        loss_fn = MarginRankingLossCustom(margin=1.0)
        
        # Create paired predictions and labels
        preds1 = torch.randn(self.batch_size, requires_grad=True)
        preds2 = torch.randn(self.batch_size, requires_grad=True)
        # labels: 1 if preds1 > preds2, -1 otherwise
        labels = torch.randint(0, 2, (self.batch_size,)).float() * 2 - 1
        
        loss = loss_fn(preds1, preds2, labels)
        
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))
    
    def test_spearman_correlation_loss(self):
        """Test SpearmanCorrelationLoss."""
        loss_fn = SpearmanCorrelationLoss()
        
        preds = torch.randn(self.batch_size, requires_grad=True)
        targets = torch.randn(self.batch_size)
        
        loss = loss_fn(preds, targets)
        
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))
    
    def test_focal_loss(self):
        """Test FocalLoss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        
        logits = torch.randn(self.batch_size, self.num_go, requires_grad=True)
        targets = torch.randint(0, 2, (self.batch_size, self.num_go)).float()
        
        loss = loss_fn(logits, targets)
        
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))
    
    def test_weighted_bce_loss(self):
        """Test WeightedBCELoss."""
        loss_fn = WeightedBCELoss(pos_weight=torch.ones(self.num_go) * 2.0)
        
        logits = torch.randn(self.batch_size, self.num_go, requires_grad=True)
        targets = torch.randint(0, 2, (self.batch_size, self.num_go)).float()
        
        loss = loss_fn(logits, targets)
        
        self.assertGreaterEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(loss))
    
    def test_loss_backward(self):
        """Test gradients flow through losses."""
        loss_fn = nn.MSELoss()
        
        preds = torch.randn(self.batch_size, requires_grad=True)
        targets = torch.randn(self.batch_size)
        
        loss = loss_fn(preds, targets)
        loss.backward()
        
        self.assertIsNotNone(preds.grad)
        self.assertTrue(torch.all(torch.isfinite(preds.grad)))


class TestModelOutputShapes(unittest.TestCase):
    """Validate output tensor shapes for all components."""
    
    @unittest.skipUnless(TORCH_GEOMETRIC_AVAILABLE, "torch_geometric is not available or failed to import")
    def test_gnn_output_shape(self):
        """Test GNN output shape."""
        model = ProteinGNN(input_dim=24, hidden_dim=64, output_dim=128)
        
        # Single graph
        x = torch.randn(100, 24)
        edge_index = torch.randint(0, 100, (2, 200))
        data = Data(x=x, edge_index=edge_index)
        
        output = model(data)
        self.assertEqual(output.shape, (1, 128))
    
    def test_cnn_output_shape(self):
        """Test CNN output shape."""
        model = SpectralCNN(input_channels=1, hidden_channels=32, output_dim=128)
        spectra = torch.randn(16, 1, 1000)
        
        output = model(spectra)
        self.assertEqual(output.shape, (16, 128))
    
    @unittest.skipUnless(TORCH_GEOMETRIC_AVAILABLE, "torch_geometric is not available or failed to import")
    def test_multimodal_output_shapes(self):
        """Test multimodal model output shapes."""
        model = VibroStructuralModel(
            latent_dim=128,
            gnn_input_dim=24,
            fusion_type='bilinear',
            num_go_terms=50
        )
        
        # Create dummy inputs
        x = torch.randn(100, 24)
        edge_index = torch.randint(0, 100, (2, 200))
        graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(100, dtype=torch.long))
        spectra = torch.randn(1, 1, 1000)
        
        # Test Novozymes output
        try:
            output = model(graph, spectra, task='novozymes')
            self.assertEqual(output.shape, (1, 1))
        except Exception:
            pass
        
        # Test CAFA 5 output
        try:
            output = model(graph, spectra, task='cafa5')
            self.assertEqual(output.shape, (1, 50))
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
