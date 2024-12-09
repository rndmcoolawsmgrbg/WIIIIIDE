import zmq
import zmq.asyncio
import asyncio
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import base64
from models import SimpleModel
import socket
import requests
from typing import Optional

logger = logging.getLogger(__name__)

class NodeService:
    def __init__(self, node_id: str, registry_address: str, port: int, external_port: Optional[int] = None):
        self.node_id = node_id
        self.port = port
        self.external_port = external_port or port  # Port to advertise to other nodes
        self.running = True
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup ZMQ sockets
        self.ctx = zmq.asyncio.Context()
        
        # Socket for job commands
        self.job_socket = self.ctx.socket(zmq.REP)
        self.job_socket.bind(f"tcp://*:{port}")
        
        # Socket for registry communication
        self.registry_socket = self.ctx.socket(zmq.REQ)
        if registry_address.startswith('tcp://'):
            registry_address = registry_address[6:]
        self.registry_socket.connect(f"tcp://{registry_address}")
        
        # Get local IP
        self.ip_address = self._get_ip_address()
        logger.info(f"Node {node_id} initialized on {self.ip_address}:{self.external_port} (device: {self.device})")
        
        self.current_job = None
        self.current_model = None
        self.optimizer = None
        
    def _get_ip_address(self) -> str:
        """Get node's local network IP address."""
        try:
            # Get local network IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Using Google's DNS to get local IP (doesn't actually connect)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            # Fallback to localhost
            return 'localhost'
            
    async def heartbeat(self):
        """Send periodic heartbeats to registry."""
        while self.running:
            try:
                msg = {
                    'type': 'node',
                    'id': self.node_id,
                    'address': f"tcp://{self.ip_address}:{self.external_port}",
                    'device': self.device,
                    'status': 'idle' if not self.current_job else 'busy'
                }
                await self.registry_socket.send_json(msg)
                await self.registry_socket.recv_json()
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
                
    async def start_training(self, config):
        """Start training with given configuration."""
        try:
            # Deserialize model class from base64 string
            model_class = pickle.loads(base64.b64decode(config['model_class'].encode('utf-8')))
            model_args = config['model_args']
            training_args = config['training_args']
            
            # Create model instance
            self.current_model = model_class(**model_args).to(self.device)
            
            # Setup optimizer
            self.optimizer = torch.optim.Adam(
                self.current_model.parameters(), 
                lr=training_args['learning_rate']
            )
            
            # Create synthetic dataset for testing
            dataset = torch.randn(1000, model_args['input_size'])
            labels = torch.randint(0, model_args['output_size'], (1000,))
            
            # Training loop
            batch_size = training_args['batch_size']
            num_epochs = training_args['num_epochs']
            
            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0
                
                # Process data in batches
                for i in range(0, len(dataset), batch_size):
                    # Get batch
                    batch_data = dataset[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    
                    # Move to device
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.current_model(batch_data)
                    loss = nn.functional.cross_entropy(outputs, batch_labels)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Log progress
                    if num_batches % 10 == 0:
                        logger.info(f"Epoch {epoch}, Batch {num_batches}, Loss: {loss.item():.4f}")
                        
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
                
            return {'status': 'completed', 'final_loss': avg_loss}
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'status': 'error', 'message': str(e)}
        
    async def handle_job(self, msg):
        """Handle incoming job request."""
        try:
            command = msg.get('command')
            if command == 'start_job':
                config = msg['config']
                logger.info(f"Starting training job with config: {config}")
                
                # Start training in background task
                self.current_job = asyncio.create_task(self.start_training(config))
                
                return {'status': 'training_started'}
                
            return {'status': 'unknown_command'}
            
        except Exception as e:
            logger.error(f"Error handling job: {e}")
            return {'status': 'error', 'message': str(e)}

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up node resources...")
        if self.current_job:
            self.current_job.cancel()
            try:
                await self.current_job
            except asyncio.CancelledError:
                pass
            
        self.job_socket.close()
        self.registry_socket.close()
        self.ctx.term()
        logger.info("Cleanup complete")

    async def run(self):
        """Main node loop."""
        heartbeat_task = None
        try:
            # Start heartbeat
            heartbeat_task = asyncio.create_task(self.heartbeat())
            
            logger.info(f"Node {self.node_id} running on port {self.port}")
            
            while self.running:
                try:
                    # Check for job messages with timeout
                    msg = await asyncio.wait_for(
                        self.job_socket.recv_json(),
                        timeout=1.0  # 1 second timeout to check running flag
                    )
                    response = await self.handle_job(msg)
                    await self.job_socket.send_json(response)
                    
                    # Update status in heartbeat
                    if self.current_job and self.current_job.done():
                        result = await self.current_job
                        logger.info(f"Training completed: {result}")
                        self.current_job = None
                        
                except asyncio.TimeoutError:
                    # Check if we should stop
                    if not self.running:
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error in job loop: {e}")
                    await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Node error: {e}")
        finally:
            self.running = False
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            await self.cleanup()