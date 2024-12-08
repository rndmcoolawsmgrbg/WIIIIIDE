import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.float()
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

class SyntheticDataset(Dataset):
    def __init__(self, size=1000, input_size=10):
        self.size = size
        self.input_size = input_size

        self.data = torch.randn(size, input_size)
        self.labels = (torch.sum(self.data, dim=1) % 10).long()
        self.attention_mask = torch.ones(size, input_size)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        } 