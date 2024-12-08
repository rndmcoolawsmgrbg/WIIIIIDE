import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pickle
import zlib
import struct

import socket
import threading
import queue
import logging
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import select
import errno

from .networking import configure_socket, send_msg, recv_msg, secure_send_msg, secure_recv_msg, perform_handshake, connect_with_retry
from .compression import FastCompression, GradientCompression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchQueue:
    """Optimized batch queue with prefetching."""
    def __init__(self, maxsize=1000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.prefetch_queue = queue.Queue(maxsize=maxsize)
        self.running = True
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        while self.running:
            try:
                batch = self.queue.get(timeout=0.1)
                self.prefetch_queue.put(batch)
            except queue.Empty:
                continue
    
    def put(self, batch):
        self.queue.put(batch)
    
    def get(self):
        try:
            return self.prefetch_queue.get_nowait()
        except queue.Empty:
            return self.queue.get()
    
    def empty(self):
        return self.queue.empty() and self.prefetch_queue.empty()
    
    def qsize(self):
        return self.queue.qsize() + self.prefetch_queue.qsize()

class CentralServer:
    def __init__(self, model, dataset, batch_size=16, ip="localhost", port=5555,
                 checkpoint_dir="checkpoints", checkpoint_interval=5, secure=False, queue_size=1000):
        """
        The class for the central server in a distributed training setup.
        
        Args:
            model: The PyTorch model to train (Required)
            dataset: The PyTorch Dataset object for training data (Required)
            batch_size: The batch size for training
            ip: The IP address to bind the server to
            port: The port to bind the server to
            checkpoint_dir: The directory to save model checkpoints
            checkpoint_interval: The interval in minutes to save checkpoints
            secure: Whether to enable secure communication

        Methods:
            start: Start the server and listen for connections
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.ip = ip
        self.port = port
        self.nodes = []
        self.batch_queue = BatchQueue(maxsize=queue_size)
        self.gradient_queue = queue.Queue()
        self.running = True
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = time.time()
        self.global_step = 0
        self.secure = secure
        self.gradient_compressor = GradientCompression()

    def distribute_batches(self):
        logger.info("Starting batch distribution...")
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        logger.info(f"Created Dataloader with {len(dataloader)} batches")
        
        while self.running:
            logger.info("Starting new epoch")
            for batch_idx, batch in enumerate(dataloader):
                if not self.running:
                    break
                
                processed_batch = {
                'batch_id': f"{self.global_step}_batch{batch_idx}",
                'input_ids': batch['input_ids'].cpu(),
                'attention_mask': batch.get('attention_mask', None).cpu() if batch.get('attention_mask') is not None else None,
                'labels': batch.get('labels', None).cpu() if batch.get('labels') is not None else None
            }
                
                logger.info(f"Putting batch {processed_batch['batch_id']} into queue")
                self.batch_queue.put(processed_batch)
                logger.info(f"Queue size: {self.batch_queue.qsize()}")
                
    def handle_node(self, conn, addr):
        """Handle node connection with minimal logging."""
        try:
            if self.secure:
                secure_conn = perform_handshake(conn, is_server=True)
            
            while self.running:
                if not self.batch_queue.empty():
                    batch = self.batch_queue.get()
                    
                    if self.secure:
                        secure_send_msg(conn, batch, secure_conn, False)
                    else:
                        send_msg(conn, batch, False)
                    
                    if self.secure:
                        gradients_data = secure_recv_msg(conn, secure_conn, False)
                    else:
                        gradients_data = recv_msg(conn, False)
                        
                    if gradients_data is None:
                        break
                    
                    if gradients_data.get('compressed', False):
                        gradients = self.gradient_compressor.decompress(
                            gradients_data['gradients'],
                            gradients_data['scales']
                        )
                    else:
                        gradients = [
                            torch.tensor(grad) if grad is not None else None 
                            for grad in gradients_data['gradients']
                        ]
                    
                    self.gradient_queue.put(gradients)
                else:
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error handling node {addr}: {e}")
        finally:
            conn.close()

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        configure_socket(server)
        server.bind((self.ip, self.port))
        server.listen(5)
        
        logger.info(f"Server started at {self.ip}:{self.port}")

        batch_thread = threading.Thread(target=self.distribute_batches, daemon=True)
        batch_thread.start()
        
        try:
            while self.running:
                conn, addr = server.accept()
                thread = threading.Thread(target=self.handle_node, args=(conn, addr))
                thread.start()
                self.nodes.append(thread)
        except KeyboardInterrupt:
            self.running = False
            server.close()

class TrainingNode:
    def __init__(self, model, server_address=('localhost', 5555), 
                 secure=False, collect_metrics=False, compress_gradients=False,
                 batch_gradients=True):
        """
        The class for a training node in a distributed training setup.

        Args:
            model: The PyTorch model to train (Required)
            server_address: The address of the central server (Default: ('localhost', 5555))
            secure: Whether to enable secure communication
            collect_metrics: Whether to collect network metrics
            compress_gradients: Whether to compress gradients
            batch_gradients: Whether to batch gradients for compression
        
        Methods:
            train: Start training the model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.server_address = server_address
        self.optimizer = torch.optim.Adam(model.parameters())
        self.running = True
        self.secure = secure
        self.collect_metrics = collect_metrics
        self.compress_gradients = compress_gradients
        self.gradient_compressor = GradientCompression() if compress_gradients else None
        self.batch_gradients = batch_gradients and compress_gradients
        logger.info(f"Using device: {self.device}")
    
    def _process_gradients(self, raw_grads, batch_id):
        """Optimized gradient processing."""
        if not self.compress_gradients:
            return {
                'batch_id': batch_id,
                'gradients': [
                    grad.cpu().numpy().tolist() if grad is not None else None  # Keep using lists
                    for grad in raw_grads
                ],
                'compressed': False
            }
        
        # Process gradients efficiently
        compressed_grads, scales = self.gradient_compressor.compress(raw_grads, batch_id)
        
        return {
            'batch_id': batch_id,
            'gradients': [
                grad.cpu().numpy().tolist() if grad is not None else None  # Keep using lists
                for grad in compressed_grads
            ],
            'scales': scales,
            'compressed': True
        }
    
    def train(self, loss_callback=None, network_callback=None):
        """
        Train the model.
        
        Args:
            loss_callback: Optional callback for loss tracking
            network_callback: Optional callback for network metrics
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_socket(sock)
        logger.info(f"Connecting to server at {self.server_address}")
        
        try:
            connect_with_retry(sock, self.server_address)
            if self.secure:
                secure_conn = perform_handshake(sock, is_server=False)
            
            while self.running:
                # Receive batch
                if self.secure:
                    result = secure_recv_msg(sock, secure_conn, self.collect_metrics)
                else:
                    result = recv_msg(sock, self.collect_metrics)
                
                if self.collect_metrics:
                    batch, recv_metrics = result
                else:
                    batch = result
                    recv_metrics = None
                
                if batch is None:
                    break

                logger.info(f"Received batch {batch['batch_id']}")

                # Process batch and compute gradients
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device) if batch.get('attention_mask') is not None else None
                labels = batch['labels'].to(self.device) if batch.get('labels') is not None else None

                self.optimizer.zero_grad()

                outputs = self.model(input_ids.float())
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                else:
                    loss = nn.CrossEntropyLoss()(outputs, input_ids.long())
                
                loss.backward()
                
                if loss_callback:
                    loss_callback(loss.item(), batch['batch_id'])
                
                # Modify the gradient sending part:
                if self.compress_gradients:
                    raw_grads = [param.grad for param in self.model.parameters()]
                    gradients_data = self._process_gradients(raw_grads, batch['batch_id'])
                else:
                    gradients_data = {
                        'batch_id': batch['batch_id'],
                        'gradients': [
                            grad.cpu().numpy().tolist() if grad is not None else None  # Keep using lists
                            for grad in [param.grad for param in self.model.parameters()]
                        ],
                        'compressed': False
                    }
                
                logger.info(f"Sending gradients for batch {batch['batch_id']}")
                if self.secure:
                    send_metrics = secure_send_msg(sock, gradients_data, secure_conn, self.collect_metrics)
                else:
                    send_metrics = send_msg(sock, gradients_data, self.collect_metrics)
                
                logger.info(f"Training loss: {loss.item():.4f}")
                
                # Report network metrics if enabled and callbacks provided
                if self.collect_metrics and network_callback and recv_metrics and send_metrics:
                    network_callback(
                        send_metrics['sent_bytes'],
                        recv_metrics['received_bytes'],
                        recv_metrics['comp_time'] + send_metrics['comp_time'],
                        recv_metrics['net_time'] + send_metrics['net_time'],
                        recv_metrics['original_size'] + send_metrics['original_size'],
                        recv_metrics['compressed_size'] + send_metrics['compressed_size']
                    )
                
        except Exception as e:
            logger.error(f"Error in training node: {e}", exc_info=True)
        finally:
            logger.info("Closing connection")
            sock.close()