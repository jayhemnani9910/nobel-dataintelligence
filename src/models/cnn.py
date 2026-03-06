"""
Spectral Encoder: 1D CNN for Vibrational Density of States

Processes 1D vibrational spectra to extract frequency-domain embeddings.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpectralCNN(nn.Module):
    """
    1D Convolutional Neural Network for vibrational spectra.
    
    Architecture:
    - Input: 1D spectrum (1000 frequency bins, 0-500 cm^-1)
    - Residual blocks with 1D convolutions
    - Adaptive pooling to fixed-size embedding
    - Output: 128-dim latent embedding
    
    Key design choices:
    - Large initial kernel (7) to capture broad spectral features
    - Max pooling to detect high-intensity modes (fingerprints)
    - Residual connections for deep networks
    """
    
    def __init__(self, input_channels: int = 1, hidden_channels: int = 32,
                 output_dim: int = 128, depth: int = 3, dropout: float = 0.1,
                 kernel_size: int = 7):
        """
        Initialize Spectral CNN.
        
        Args:
            input_channels: Number of input channels (1 for single spectrum)
            hidden_channels: Base hidden dimension
            output_dim: Output embedding dimension
            depth: Number of residual blocks
            dropout: Dropout probability
            kernel_size: Initial convolution kernel size
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        
        # Entry block
        self.entry = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=kernel_size,
                     stride=2, padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(depth):
            in_channels = hidden_channels * (2**i)
            out_channels = hidden_channels * (2**(i+1))
            
            self.residual_blocks.append(
                ResidualBlock1D(in_channels, out_channels, dropout=dropout)
            )
        
        # Global adaptive pooling
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        
        # Projection to output dimension
        final_channels = hidden_channels * (2**depth)
        self.output_proj = nn.Sequential(
            nn.Linear(final_channels, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        logger.info(f"Initialized SpectralCNN with {depth} residual blocks")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral encoder.
        
        Args:
            x: Input spectra (batch_size, 1, 1000)
            
        Returns:
            Embeddings (batch_size, output_dim)
        """
        # Entry block
        x = self.entry(x)  # (B, hidden, 500)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Global pooling
        x = self.adaptive_pool(x)  # (B, final_channels, 1)
        x = x.squeeze(-1)  # (B, final_channels)
        
        # Output projection
        embedding = self.output_proj(x)  # (B, output_dim)
        
        return embedding


class ResidualBlock1D(nn.Module):
    """
    1D Residual block with batch normalization and ReLU.
    
    Architecture:
    Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> Add -> ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = 0.1):
        """
        Initialize 1D residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Reduce resolution via strided convolution
        stride = 2 if in_channels != out_channels else 1
        
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                     padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
        )
        
        # Projection for skip connection if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.skip(x)
        out = self.main(x)
        out = out + identity
        out = F.relu(out)
        return out


class SpectralFeatureExtractor(nn.Module):
    """
    Extract handcrafted features from spectra before neural processing.
    
    Features:
    - Integral (total intensity)
    - Peak height
    - Peak frequency
    - Spectral centroid
    - Number of peaks above threshold
    """
    
    def __init__(self, n_points: int = 1000, freq_max: float = 500.0):
        """
        Initialize feature extractor.
        
        Args:
            n_points: Number of frequency points
            freq_max: Maximum frequency (cm^-1)
        """
        super().__init__()
        
        self.n_points = n_points
        self.freq_max = freq_max
        self.freq_axis = torch.linspace(0, freq_max, n_points)
    
    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        """
        Extract features from batch of spectra.
        
        Args:
            spectra: (batch_size, 1000)
            
        Returns:
            Features: (batch_size, n_features)
        """
        batch_size = spectra.size(0)
        device = spectra.device
        
        # Move frequency axis to device
        freq_axis = self.freq_axis.to(device)
        
        features = []
        for spec in spectra:
            spec = spec.squeeze()  # Remove channel dim if present
            
            # Integral
            integral = torch.sum(spec)
            
            # Peak height and frequency
            peak_height = torch.max(spec)
            peak_idx = torch.argmax(spec)
            peak_freq = freq_axis[peak_idx]
            
            # Centroid
            centroid = torch.sum(freq_axis * spec) / torch.sum(spec)
            
            # Std dev
            std_dev = torch.sqrt(
                torch.sum((freq_axis - centroid)**2 * spec) / torch.sum(spec)
            )
            
            # Number of peaks (threshold = 0.1 * max)
            threshold = 0.1 * peak_height
            above_threshold = torch.sum(spec > threshold).float()
            
            spec_features = torch.stack([
                integral,
                peak_height,
                peak_freq,
                centroid,
                std_dev,
                above_threshold
            ])
            
            features.append(spec_features)
        
        return torch.stack(features)  # (batch_size, 6)


class MultiScaleSpectralCNN(nn.Module):
    """
    Multi-scale variant of SpectralCNN that processes spectra
    at different resolutions and fuses features.
    
    Useful for capturing both local peaks and global spectral shape.
    """
    
    def __init__(self, input_channels: int = 1, hidden_channels: int = 32,
                 output_dim: int = 128, dropout: float = 0.1):
        """Initialize multi-scale spectral CNN."""
        super().__init__()
        
        # Multiple pathways at different scales
        # Multi-scale branches
        # Each branch produces output_dim // 3, and we concatenate 3 of them
        branch_dim = output_dim // 3
        self.scale1 = SpectralCNN(input_channels, hidden_channels, branch_dim)
        self.scale2 = SpectralCNN(input_channels, hidden_channels, branch_dim)
        self.scale3 = SpectralCNN(input_channels, hidden_channels, branch_dim)
        
        # Fusion layer: input is 3 * branch_dim (which may not equal output_dim due to integer division)
        fused_dim = branch_dim * 3
        self.fusion = nn.Linear(fused_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale architecture."""
        # Apply max pooling to create different resolutions
        x1 = x
        x2 = F.max_pool1d(x, kernel_size=2, stride=2)
        x3 = F.max_pool1d(x, kernel_size=4, stride=4)
        
        # Process at different scales
        e1 = self.scale1(x1)
        e2 = self.scale2(x2)
        e3 = self.scale3(x3)
        
        # Concatenate and fuse
        fused = torch.cat([e1, e2, e3], dim=-1)
        embedding = self.fusion(fused)
        
        return embedding


def main():
    """Test SpectralCNN module."""
    logger.info("Testing SpectralCNN module...")
    
    # Create dummy spectral data
    batch_size = 16
    n_freqs = 1000
    
    # Synthetic spectra (batch_size, 1, n_freqs)
    spectra = torch.randn(batch_size, 1, n_freqs)
    
    # Initialize model
    model = SpectralCNN(input_channels=1, hidden_channels=32, output_dim=128)
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(spectra)
    
    logger.info(f"Input shape: {spectra.shape}")
    logger.info(f"Output embedding shape: {embeddings.shape}")
    assert embeddings.shape == (batch_size, 128), "Unexpected output shape"
    
    # Test feature extraction
    logger.info("\nTesting SpectralFeatureExtractor...")
    feature_extractor = SpectralFeatureExtractor(n_points=n_freqs)
    features = feature_extractor(spectra.squeeze(1))
    logger.info(f"Extracted features shape: {features.shape}")
    
    logger.info("SpectralCNN tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
