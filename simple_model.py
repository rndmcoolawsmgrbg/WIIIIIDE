import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SimpleModel(nn.Module):
    def __init__(self, input_size=1024, hidden_size=4096, output_size=1024):
        super().__init__()
        # Simple feed-forward layers with large dimensions
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # Basic dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Ensure we're working with float tensors
        x = x.float()  # [batch, seq_len, input_size]
        
        # Process each token in the sequence
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)  # [batch, seq_len, output_size]
        
        # Average over sequence length to get one prediction per batch
        x = torch.mean(x, dim=1)  # [batch, output_size]
        return x

class SyntheticDataset(Dataset):
    def __init__(self, size=1000, input_size=1024, sequence_length=128, num_classes=1024):
        self.size = size
        self.input_size = input_size
        self.sequence_length = sequence_length

        # Create large tensors for synthetic data
        self.data = torch.randn(size, sequence_length, input_size)
        # Create labels with fewer classes for better training
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx]
        } 