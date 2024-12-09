import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import threading
import queue
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import lz4.frame
import asyncio
import zmq
import zmq.asyncio
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkManager:
    """Efficient network communication with adaptive compression."""
    
    def __init__(self, compression_threshold: int = 1024 * 50):
        self.compression_threshold = compression_threshold
        self.ctx = zmq.asyncio.Context()
        
    async def send_tensor(self, socket, tensor: torch.Tensor, identity: bytes = None) -> None:
        """Efficiently send tensor with optional compression."""
        # Move to CPU and get numpy array
        tensor_np = tensor.detach().cpu().numpy()
        
        # Get metadata
        metadata = {
            'shape': tensor_np.shape,
            'dtype': str(tensor_np.dtype),
            'compressed': False
        }
        
        # Convert to bytes
        data = tensor_np.tobytes()
        
        # Compress if large enough
        if len(data) > self.compression_threshold:
            data = lz4.frame.compress(data)
            metadata['compressed'] = True
        
        # Prepare message parts
        metadata_bytes = pickle.dumps(metadata)
        message = [b'tensor', metadata_bytes, data]
        if identity:
            message = [identity] + message
        await socket.send_multipart(message)
        
    async def recv_tensor(self, socket) -> torch.Tensor:
        """Receive tensor with automatic decompression."""
        msg = await socket.recv_multipart()
        
        # Handle message parts
        if len(msg) == 4:  # With identity
            _, msg_type, metadata_bytes, data = msg
        else:  # Without identity
            msg_type, metadata_bytes, data = msg
            
        metadata = pickle.loads(metadata_bytes)
        
        # Decompress if needed
        if metadata['compressed']:
            data = lz4.frame.decompress(data)
            
        # Convert back to tensor
        array = np.frombuffer(data, dtype=np.dtype(metadata['dtype']))
        array = array.reshape(metadata['shape'])
        return torch.from_numpy(array)

class DistributedServer:
    """Coordinates distributed training across workers."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        port: int = 5555,
        world_size: int = 2
    ):
        self.model = model
        self.optimizer = optimizer
        self.port = port
        self.world_size = world_size
        self.running = True
        
        # Network setup
        self.network = NetworkManager()
        self.socket = self.network.ctx.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://*:{port}")
        
        # Training state
        self.gradient_accumulator = {}
        self.batch_counter = 0
        self.gradient_lock = asyncio.Lock()
        self.message_queues = {}
        self.connected_workers = set()
        
    def _get_worker_name(self, worker_id: bytes) -> str:
        """Convert worker ID bytes to readable name."""
        try:
            return worker_id.decode()
        except:
            return f"worker_{worker_id.hex()[:6]}"

    async def handle_model_request(self, worker_id: bytes):
        """Handle model parameter request."""
        worker_name = self._get_worker_name(worker_id)
        logger.info(f"Sending model to {worker_name}")
        for param in self.model.parameters():
            await self.network.send_tensor(self.socket, param.data, worker_id)
        logger.info(f"Model sent to {worker_name}")
        
    async def handle_gradients(self, worker_id: bytes, batch_id: int):
        """Handle incoming gradients."""
        worker_name = self._get_worker_name(worker_id)
        async with self.gradient_lock:
            if batch_id not in self.gradient_accumulator:
                self.gradient_accumulator[batch_id] = []
                
            grads = []
            for param in self.model.parameters():
                grad = await self.network.recv_tensor(self.socket)
                grads.append(grad)
                
            self.gradient_accumulator[batch_id].append(grads)
            logger.info(f"Received gradients from {worker_name} for batch {batch_id}")
            
            if len(self.gradient_accumulator[batch_id]) == self.world_size:
                await self._update_model(batch_id)
                
    async def _update_model(self, batch_id: int):
        """Apply accumulated gradients to model."""
        try:
            # Average gradients
            averaged_grads = []
            for param_idx in range(len(list(self.model.parameters()))):
                grads = [
                    acc_grads[param_idx] 
                    for acc_grads in self.gradient_accumulator[batch_id]
                ]
                avg_grad = torch.stack(grads).mean(dim=0)
                averaged_grads.append(avg_grad)
            
            # Apply gradients
            for param, grad in zip(self.model.parameters(), averaged_grads):
                param.grad = grad
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            logger.info(f"Applied averaged gradients for batch {batch_id}")
            
            # Cleanup
            del self.gradient_accumulator[batch_id]
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            
    async def message_handler(self):
        """Central message handling loop."""
        while self.running:
            try:
                msg = await self.socket.recv_multipart()
                worker_id, command = msg[0], msg[1]
                worker_name = self._get_worker_name(worker_id)
                
                if command == b'hello':
                    if worker_id not in self.connected_workers:
                        logger.info(f"New worker connected: {worker_name}")
                        self.connected_workers.add(worker_id)
                        await self.socket.send_multipart([worker_id, b'welcome'])
                        
                elif command == b'get_model':
                    await self.handle_model_request(worker_id)
                    
                elif command == b'gradients':
                    batch_id = int(msg[2])
                    await self.handle_gradients(worker_id, batch_id)
                    
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                
    async def serve(self):
        """Main server loop."""
        logger.info(f"Server listening on port {self.port}")
        await self.message_handler()
                
    def start(self):
        """Start server in asyncio event loop."""
        asyncio.run(self.serve())

class DistributedWorker:
    """Worker node for distributed training."""
    
    def __init__(
        self,
        model: nn.Module,
        server_address: str,
        worker_id: str = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.server_address = server_address
        self.device = device or self._get_device()
        self.model.to(self.device)
        self.worker_id = worker_id or "worker"
        
        # Network setup
        self.network = NetworkManager()
        self.socket = self.network.ctx.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{server_address}")
        self.socket.identity = self.worker_id.encode()
        logger.info(f"Connected to {server_address} as {self.worker_id}")

    @classmethod
    async def create(cls, model: nn.Module, server_address: str, worker_id: str = None, device: Optional[str] = None):
        """Async factory method to create and initialize worker."""
        worker = cls(model, server_address, worker_id, device)
        try:
            # Send initial handshake
            await worker.socket.send_multipart([b'hello'])
            response = await worker.socket.recv_multipart()
            if response[0] != b'welcome':
                raise ConnectionError("Failed to establish connection with server")
            
            # Test connection with initial sync
            await worker.sync_model()
            logger.info(f"Worker {worker.worker_id} initialized and synced")
            return worker
        except Exception as e:
            logger.error(f"Failed to initialize worker {worker_id}: {e}")
            raise

    def _get_device(self) -> str:
        """Get best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
        
    async def sync_model(self):
        """Get latest model state from server."""
        try:
            logger.info("Requesting model update")
            await self.socket.send_multipart([b'get_model'])
            
            # Receive and load parameters
            logger.info("Receiving model parameters")
            for param in self.model.parameters():
                tensor = await self.network.recv_tensor(self.socket)
                param.data.copy_(tensor)
            logger.info("Model parameters updated")
        except Exception as e:
            logger.error(f"Error during model sync: {e}")
            raise
        
    async def train_batch(self, batch: Dict[str, torch.Tensor], batch_id: int):
        """Train on a single batch and send gradients."""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(batch['input_ids'])
        loss = nn.functional.cross_entropy(outputs, batch['labels'])
        
        # Backward pass
        loss.backward()
        
        # Send gradients
        await self.socket.send_multipart([b'gradients', str(batch_id).encode()])
        for param in self.model.parameters():
            if param.grad is not None:
                await self.network.send_tensor(self.socket, param.grad)
            else:
                await self.network.send_tensor(self.socket, torch.zeros_like(param))
                
        return loss.item() 