import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle, zlib, struct, socket, threading
import queue, logging, time, os, select, errno
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .networking import (
    configure_socket, send_msg, recv_msg, 
    secure_send_msg, secure_recv_msg, 
    perform_handshake, connect_with_retry,
    configure_server_socket
)
from .compression import FastCompression, GradientCompression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device():
    """Determine the available computing device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BatchQueue:
    """Optimized batch queue with prefetching capabilities."""
    
    def __init__(self, maxsize=1000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.prefetch_queue = queue.Queue(maxsize=maxsize)
        self.running = True
        
        # Start prefetch worker thread
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker, 
            daemon=True
        )
        self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        """Background worker to prefetch batches."""
        while self.running:
            try:
                batch = self.queue.get(timeout=0.1)
                self.prefetch_queue.put(batch)
            except queue.Empty:
                continue
    
    def put(self, batch):
        """Add a batch to the queue."""
        self.queue.put(batch)
    
    def get(self):
        """Retrieve a batch, preferably from prefetch queue."""
        try:
            return self.prefetch_queue.get_nowait()
        except queue.Empty:
            return self.queue.get()
    
    def empty(self):
        """Check if both queues are empty."""
        return self.queue.empty() and self.prefetch_queue.empty()
    
    def qsize(self):
        """Get total size of both queues."""
        return self.queue.qsize() + self.prefetch_queue.qsize()

class CentralServer:
    """Central server for distributed training coordination."""
    
    def __init__(
        self, 
        model, 
        dataset, 
        batch_size=16, 
        ip="localhost", 
        port=5555,
        checkpoint_dir="checkpoints", 
        checkpoint_interval=5, 
        secure=False, 
        queue_size=1000,
        criterion=None
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.ip = ip
        self.port = port
        self.nodes = []
        self.batch_queue = BatchQueue(maxsize=queue_size)
        self.gradient_queue = queue.Queue()
        self.running = True
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = time.time()
        self.global_step = 0

        self.secure = secure
        self.gradient_compressor = GradientCompression()

    def distribute_batches(self):
        """Distribute batches to worker nodes."""
        logger.info("Starting batch distribution...")
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        batch_count = 0
        while self.running:
            for batch_idx, batch in enumerate(dataloader):
                if not self.running:
                    break
                
                processed_batch = {
                    'batch_id': f"{self.global_step}_batch{batch_idx}",
                    'input_ids': batch['input_ids'].cpu(),
                    'attention_mask': batch.get('attention_mask', None),
                    'labels': batch.get('labels', None)
                }
                
                self.batch_queue.put(processed_batch)
                batch_count += 1
                
    def handle_node(self, conn, addr):
        """Handle node connection with detailed logging."""
        logger.info(f"New node connected from {addr}")
        try:
            if self.secure:
                secure_conn = perform_handshake(conn, is_server=True)
                logger.info(f"Completed secure handshake with {addr}")
            
            while self.running:
                if not self.batch_queue.empty():
                    batch = self.batch_queue.get()
                    logger.info(f"Sending batch {batch['batch_id']} to {addr}")
                    
                    try:
                        if self.secure:
                            secure_send_msg(conn, batch, secure_conn, False)
                        else:
                            send_msg(conn, batch, False)
                        logger.info(f"Successfully sent batch to {addr}")

                        logger.info(f"Waiting for gradients from {addr}")
                        if self.secure:
                            gradients_data = secure_recv_msg(conn, secure_conn, False)
                        else:
                            gradients_data = recv_msg(conn, False)
                            
                        if gradients_data is None:
                            logger.warning(f"Received None gradients from {addr}")
                            break
                        
                        logger.info(f"Received gradients for batch {gradients_data.get('batch_id')} from {addr}")
                        
                    except Exception as e:
                        logger.error(f"Error in communication with {addr}: {e}")
                        break
                else:
                    time.sleep(0.1)  # Prevent busy waiting

        except Exception as e:
            logger.error(f"Error handling node {addr}: {e}", exc_info=True)
        finally:
            logger.info(f"Closing connection with {addr}")
            conn.close()

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        configure_server_socket(server)
        server.bind((self.ip, self.port))
        server.listen(5)
        
        logger.info(f"Server started at {self.ip}:{self.port}")

        # Start batch distribution in a separate thread
        batch_thread = threading.Thread(target=self.distribute_batches, daemon=True)
        batch_thread.start()
        logger.info("Batch distribution thread started")
        
        try:
            while self.running:
                try:
                    # Non-blocking accept with select
                    readable, _, _ = select.select([server], [], [], 1.0)  # 1 second timeout
                    if readable:
                        conn, addr = server.accept()
                        configure_socket(conn)  # Configure client connection
                        logger.info(f"Accepted connection from {addr}")
                        thread = threading.Thread(target=self.handle_node, args=(conn, addr))
                        thread.start()
                        self.nodes.append(thread)
                        logger.info(f"Started handler thread for {addr}")
                except socket.error as e:
                    if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                        logger.error(f"Socket error: {e}")
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    continue
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.running = False
        finally:
            logger.info("Closing server")
            server.close()

class TrainingNode:
    """Worker node for distributed training."""
    
    def __init__(
        self, 
        model, 
        server_address=('localhost', 5555), 
        secure=False, 
        collect_metrics=False, 
        compress_gradients=False,
        device=None,
        criterion=None,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.server_address = server_address
        self.secure = secure
        self.collect_metrics = collect_metrics
        self.compress_gradients = compress_gradients
        self.gradient_compressor = GradientCompression() if compress_gradients else None
        self.socket = None
        self.secure_conn = None
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        
    def connect(self):
        """Establish connection with the central server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        configure_socket(self.socket)
        logger.info(f"Connecting to server at {self.server_address}")
        
        connect_with_retry(self.socket, self.server_address)
        if self.secure:
            self.secure_conn = perform_handshake(self.socket, is_server=False)
        logger.info("Connected to server.")
        return self
    
    def get_batch(self):
        """
        Receive a batch from the server.
        
        Returns:
            tuple: (batch_data, metrics) if collect_metrics=True
            dict: batch_data if collect_metrics=False
        """
        if self.secure:
            result = secure_recv_msg(self.socket, self.secure_conn, self.collect_metrics)
        else:
            result = recv_msg(self.socket, self.collect_metrics)
        
        return result
    
    def send_gradients(self, gradients, batch_id):
        """
        Send computed gradients back to the server.
        
        Args:
            gradients: List of gradient tensors
            batch_id: ID of the processed batch
            
        Returns:
            dict: Metrics if collect_metrics=True
            None: if collect_metrics=False
        """
        if self.compress_gradients:
            gradients_data = self._process_gradients(gradients, batch_id)
        else:
            gradients_data = {
                'batch_id': batch_id,
                'gradients': [
                    grad.cpu().numpy().tolist() if grad is not None else None
                    for grad in gradients
                ],
                'compressed': False
            }
        
        if self.secure:
            return secure_send_msg(self.socket, gradients_data, self.secure_conn, self.collect_metrics)
        else:
            return send_msg(self.socket, gradients_data, self.collect_metrics)
    
    def _process_gradients(self, raw_grads, batch_id):
        """Process and optionally compress gradients."""
        if not self.compress_gradients:
            return {
                'batch_id': batch_id,
                'gradients': [
                    grad.cpu().numpy().tolist() if grad is not None else None
                    for grad in raw_grads
                ],
                'compressed': False
            }
        
        compressed_grads, scales = self.gradient_compressor.compress(raw_grads, batch_id)
        return {
            'batch_id': batch_id,
            'gradients': [
                grad.cpu().numpy().tolist() if grad is not None else None
                for grad in compressed_grads
            ],
            'scales': scales,
            'compressed': True
        }
    
    def close(self):
        """Close the connection with the server."""
        if self.socket:
            self.socket.close()
            self.socket = None
            self.secure_conn = None

    def train(self, loss_callback=None, network_callback=None):
        """Train the model using batches from the server."""
        try:
            self.connect()
            optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

            while True:
                try:
                    result = self.get_batch()
                    
                    if result is None:
                        logger.warning("Received None result from get_batch")
                        break
                    
                    if self.collect_metrics:
                        batch, metrics = result
                        if batch is None:
                            logger.warning("Received None batch with metrics")
                            break
                        
                        if network_callback and metrics:
                            network_callback(
                                sent_bytes=metrics.get('sent_bytes', 0),
                                received_bytes=metrics.get('received_bytes', 0),
                                comp_time=metrics.get('comp_time', 0),
                                net_time=metrics.get('net_time', 0),
                                original_size=metrics.get('original_size', 0),
                                compressed_size=metrics.get('compressed_size', 0)
                            )
                    else:
                        batch = result
                        if batch is None:
                            logger.warning("Received None batch")
                            break

                    # Process batch
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch.get('attention_mask')
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                    labels = batch.get('labels')
                    if labels is not None:
                        labels = labels.to(self.device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(input_ids)
                    
                    # Calculate loss
                    if labels is not None:
                        loss = self.criterion(outputs, labels)
                    else:
                        loss = self.criterion(outputs, input_ids.long())

                    # Backward pass
                    loss.backward()

                    # Report loss if callback provided
                    if loss_callback:
                        loss_callback(loss.item(), batch['batch_id'])

                    # Get gradients
                    gradients = [param.grad for param in self.model.parameters()]
                    
                    # Send gradients back to server
                    metrics = self.send_gradients(gradients, batch['batch_id'])
                    
                    if self.collect_metrics and network_callback and metrics:
                        network_callback(
                            sent_bytes=metrics.get('sent_bytes', 0),
                            received_bytes=metrics.get('received_bytes', 0),
                            comp_time=metrics.get('comp_time', 0),
                            net_time=metrics.get('net_time', 0),
                            original_size=metrics.get('original_size', 0),
                            compressed_size=metrics.get('compressed_size', 0)
                        )

                except Exception as e:
                    logger.error(f"Error in training iteration: {e}")
                    break

        except Exception as e:
            logger.error(f"Error in training loop: {e}", exc_info=True)
            raise
        finally:
            self.close()