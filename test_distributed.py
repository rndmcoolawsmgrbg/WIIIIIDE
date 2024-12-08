import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from w5xde import CentralServer, TrainingNode
import threading
import time

# Simple synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, size=1000, input_size=10):
        self.size = size
        self.input_size = input_size
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx]
        }

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=5, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def run_server(model, dataset):
    server = CentralServer(
        model=model,
        dataset=dataset,
        batch_size=32,
        ip="localhost",
        port=5555
    )
    server.start()

def run_worker(model, worker_id):
    def loss_callback(loss, batch_id):
        print(f"Worker {worker_id} - Batch {batch_id}: Loss = {loss:.4f}")

    node = TrainingNode(
        model=model,
        server_address=('localhost', 5555),
        collect_metrics=True
    )
    node.train(loss_callback=loss_callback)

def main():
    # Create synthetic dataset
    dataset = SyntheticDataset(size=1000, input_size=10)
    
    # Create model
    model = SimpleModel(input_size=10, hidden_size=5, num_classes=2)
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=run_server,
        args=(model, dataset),
        daemon=True
    )
    server_thread.start()
    
    # Wait for server to initialize
    time.sleep(2)
    
    # Create workers
    num_workers = 2
    workers = []
    for i in range(num_workers):
        worker_model = SimpleModel(input_size=10, hidden_size=5, num_classes=2)
        worker_thread = threading.Thread(
            target=run_worker,
            args=(worker_model, i),
            daemon=True
        )
        workers.append(worker_thread)
        worker_thread.start()
    
    # Wait for workers to complete
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    main() 