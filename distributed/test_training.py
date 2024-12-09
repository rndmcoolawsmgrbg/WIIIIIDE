import torch
import torch.nn as nn
import asyncio
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
from core import DistributedServer, DistributedWorker
from typing import Dict, List
import logging
import threading
from collections import deque
import psutil
import GPUtil
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

# Synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, size=10000, input_size=784, num_classes=10):
        self.data = torch.randn(size, input_size)
        self.labels = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx]
        }

class MetricsTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.losses = deque(maxlen=window_size)
        self.batch_times = deque(maxlen=window_size)
        self.network_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.total_batches = 0
        
    def update(self, loss: float, batch_time: float, network_time: float):
        self.losses.append(loss)
        self.batch_times.append(batch_time)
        self.network_times.append(network_time)
        self.total_batches += 1
        
    def get_metrics(self) -> Dict[str, float]:
        elapsed = time.time() - self.start_time
        return {
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'avg_network_time': np.mean(self.network_times) if self.network_times else 0,
            'batches_per_second': len(self.batch_times) / elapsed if elapsed > 0 else 0,
            'total_batches': self.total_batches,
            'gpu_util': GPUtil.getGPUs()[0].load if GPUtil.getGPUs() else 0,
            'cpu_util': psutil.cpu_percent(),
            'memory_util': psutil.virtual_memory().percent
        }

    def print_summary(self):
        metrics = self.get_metrics()
        logger.info("=== Training Summary ===")
        logger.info(f"Total batches processed: {metrics['total_batches']}")
        logger.info(f"Average loss: {metrics['avg_loss']:.4f}")
        logger.info(f"Average batch time: {metrics['avg_batch_time']*1000:.2f}ms")
        logger.info(f"Average network time: {metrics['avg_network_time']*1000:.2f}ms")
        logger.info(f"Training speed: {metrics['batches_per_second']:.2f} batches/s")
        logger.info(f"CPU utilization: {metrics['cpu_util']:.1f}%")
        logger.info(f"Memory utilization: {metrics['memory_util']:.1f}%")
        if GPUtil.getGPUs():
            logger.info(f"GPU utilization: {metrics['gpu_util']*100:.1f}%")

async def run_worker(worker_id: int, server_address: str, dataset: Dataset, metrics: MetricsTracker, max_batches: int = 1000):
    logger.info(f"Starting worker {worker_id}")
    try:
        # Initialize worker
        model = SimpleModel()
        logger.info(f"Worker {worker_id} model created")
        
        try:
            # Use the async factory method with worker_id
            worker = await DistributedWorker.create(
                model, 
                server_address,
                worker_id=f"worker_{worker_id}"  # Add human-readable ID
            )
            logger.info(f"Worker {worker_id} connected to server")
        except Exception as e:
            logger.error(f"Failed to initialize worker {worker_id}: {e}")
            return
        
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=True,
            num_workers=0  # Use single process loading for testing
        )
        logger.info(f"Worker {worker_id} dataloader created")
        
        # Training loop
        batch_id = 0
        sync_time = 0
        
        while batch_id < max_batches:
            try:
                # Sync with server periodically
                if batch_id % 10 == 0:
                    start_sync = time.time()
                    await worker.sync_model()
                    sync_time = time.time() - start_sync
                    logger.info(f"Worker {worker_id} synced with server (took {sync_time*1000:.2f}ms)")
                
                for batch in dataloader:
                    if batch_id >= max_batches:
                        break
                        
                    start_batch = time.time()
                    
                    try:
                        # Train batch
                        loss = await worker.train_batch(batch, batch_id)
                        
                        # Update metrics
                        batch_time = time.time() - start_batch
                        metrics.update(loss, batch_time, sync_time if batch_id % 10 == 0 else 0)
                        
                        # Log metrics every 10 batches
                        if batch_id % 10 == 0:
                            current_metrics = metrics.get_metrics()
                            logger.info(
                                f"Worker {worker_id} - "
                                f"Batch {batch_id}: Loss = {current_metrics['avg_loss']:.4f}, "
                                f"Speed = {current_metrics['batches_per_second']:.2f} batches/s"
                            )
                        
                        batch_id += 1
                    except Exception as e:
                        logger.error(f"Error processing batch {batch_id}: {e}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                break
                
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
    finally:
        logger.info(f"Worker {worker_id} finished")

def start_server(model: nn.Module, world_size: int):
    optimizer = torch.optim.Adam(model.parameters())
    server = DistributedServer(model, optimizer, world_size=world_size)
    server.start()

class DistributedTrainer:
    def __init__(self):
        self.running = True
        self.metrics = MetricsTracker()
        signal.signal(signal.SIGINT, self.handle_interrupt)
        
    def handle_interrupt(self, signum, frame):
        logger.info("Shutting down gracefully...")
        self.running = False
        self.metrics.print_summary()
        sys.exit(0)
        
    async def run(self):
        # Parameters
        world_size = 1
        dataset_size = 100000
        input_size = 784
        num_classes = 10
        max_batches = 1000  # Limit total batches for testing
        
        logger.info("Initializing distributed training...")
        
        # Create dataset
        dataset = SyntheticDataset(dataset_size, input_size, num_classes)
        logger.info(f"Created synthetic dataset with {dataset_size} samples")
        
        # Start server in separate thread
        model = SimpleModel()
        server_thread = threading.Thread(
            target=start_server,
            args=(model, world_size)
        )
        server_thread.start()
        logger.info("Server started")
        
        # Wait for server to start and be ready
        await asyncio.sleep(5)  # Increased wait time
        logger.info("Server should be ready")
        
        # Create and start workers
        workers = []
        logger.info(f"Starting {world_size} workers...")
        for i in range(world_size):
            worker = asyncio.create_task(
                run_worker(i, "localhost:5555", dataset, self.metrics, max_batches)
            )
            workers.append(worker)
            await asyncio.sleep(1)  # Add delay between worker starts
        
        # Wait for workers
        try:
            await asyncio.gather(*workers)
        except Exception as e:
            logger.error(f"Error in worker coordination: {e}", exc_info=True)
        finally:
            self.metrics.print_summary()

if __name__ == "__main__":
    trainer = DistributedTrainer()
    asyncio.run(trainer.run()) 