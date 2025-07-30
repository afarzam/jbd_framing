import torch
import torch.nn as nn
from typing import Optional

class JailbreakDetector(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        hidden_dims: list = [1024, 512, 256],
        dropout_rate: float = 0.2
    ):
        """
        Initialize jailbreak detector model that takes LLM encoder representations as input
        
        Args:
            encoder_dim: Dimension of the LLM encoder output
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
        """
        super(JailbreakDetector, self).__init__()
        
        # Build sequential layers
        layers = []
        prev_dim = encoder_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Add final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, encoder_dim)
                containing LLM encoder representations
            
        Returns:
            Output tensor of shape (batch_size, 1) containing logits
        """
        return self.classifier(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions with probability threshold
        
        Args:
            x: Input tensor of shape (batch_size, encoder_dim)
                containing LLM encoder representations
            threshold: Probability threshold for classification
            
        Returns:
            Binary predictions tensor of shape (batch_size,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).long().squeeze() 