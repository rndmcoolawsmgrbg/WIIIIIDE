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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_msg(sock, msg, collect_metrics=False):
    """Optimized message sending with faster compression."""
    if collect_metrics:
        start_comp = time.time()
    
    # Use pickle protocol 5 for better performance
    msg_bytes = pickle.dumps(msg, protocol=5)
    original_size = len(msg_bytes) if collect_metrics else 0
    
    # Use highest compression level for first message (model architecture)
    # Use fast compression for gradients
    if isinstance(msg, dict) and 'gradients' in msg:
        # For gradients, use fast compression
        compressed_msg = zlib.compress(msg_bytes, level=1)
    else:
        # For model architecture, use best compression
        compressed_msg = zlib.compress(msg_bytes, level=9)
    
    compressed_size = len(compressed_msg) if collect_metrics else 0
    
    if collect_metrics:
        comp_time = time.time() - start_comp
        start_net = time.time()
    
    # Send length and data in a single call when possible
    msg_len = len(compressed_msg)
    header = struct.pack(">I", msg_len)
    
    try:
        # Use sendall with combined buffer for fewer syscalls
        sock.sendall(header + compressed_msg)
    except BlockingIOError:
        # Fall back to separate sends if buffer is full
        sock.sendall(header)
        sock.sendall(compressed_msg)
    
    if collect_metrics:
        net_time = time.time() - start_net
        return {
            'sent_bytes': msg_len + 4,
            'received_bytes': 0,
            'comp_time': comp_time,
            'net_time': net_time,
            'original_size': original_size,
            'compressed_size': compressed_size
        }
    return None

def recv_msg(sock, collect_metrics=False):
    """Optimized message receiving."""
    if collect_metrics:
        start_net = time.time()
    
    # Receive length
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return (None, None) if collect_metrics else None
    
    msglen = struct.unpack('>I', raw_msglen)[0]
    
    # Receive data
    compressed_msg = recvall(sock, msglen)
    if compressed_msg is None:
        return (None, None) if collect_metrics else None
    
    if collect_metrics:
        net_time = time.time() - start_net
        start_comp = time.time()
    
    # Decompress and unpickle
    msg_bytes = zlib.decompress(compressed_msg)
    msg = pickle.loads(msg_bytes)
    
    if collect_metrics:
        comp_time = time.time() - start_comp
        metrics = {
            'sent_bytes': 0,
            'received_bytes': msglen + 4,
            'comp_time': comp_time,
            'net_time': net_time,
            'original_size': len(msg_bytes),
            'compressed_size': msglen
        }
        return msg, metrics
    return msg

def generate_key():
    return base64.urlsafe_b64encode(os.urandom(32))

class SecureConnection:
    def __init__(self):
        self.fernet = None
    
    def create_key(self, shared_secret):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'static_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(shared_secret))
        self.fernet = Fernet(key)

    def encrypt_message(self, msg):
        if not isinstance(msg, bytes):
            msg = bytes(msg)
        return self.fernet.encrypt(msg)

    def decrypt_message(self, encrypted_msg):
        if isinstance(encrypted_msg, bytearray):
            encrypted_msg = bytes(encrypted_msg)
        return self.fernet.decrypt(encrypted_msg)

def secure_send_msg(sock, msg, secure_conn, collect_metrics=False):
    try:
        start_comp = time.time()
        pickled_msg = pickle.dumps(msg)
        original_size = len(pickled_msg)
        compressed_msg = zlib.compress(pickled_msg)
        compressed_size = len(compressed_msg)
        encrypted_msg = secure_conn.encrypt_message(compressed_msg)
        comp_time = time.time() - start_comp
        
        start_net = time.time()
        msg_len = len(encrypted_msg)
        sock.sendall(struct.pack(">I", msg_len))
        sock.sendall(encrypted_msg)
        net_time = time.time() - start_net
        
        if collect_metrics:
            return {
                'sent_bytes': msg_len + 4,  # Include length header
                'received_bytes': 0,
                'comp_time': comp_time,
                'net_time': net_time,
                'original_size': original_size,
                'compressed_size': compressed_size
            }
        return None
        
    except Exception as e:
        logger.error(f"Error in secure_send_msg: {e}")
        raise

def secure_recv_msg(sock, secure_conn, collect_metrics=False):
    try:
        start_net = time.time()
        raw_msglen = recvall(sock, 4)
        if not raw_msglen:
            return None, None
        msglen = struct.unpack('>I', raw_msglen)[0]
        encrypted_msg = recvall(sock, msglen)
        if encrypted_msg is None:
            return None, None
        net_time = time.time() - start_net
        
        start_comp = time.time()
        decrypted_msg = secure_conn.decrypt_message(encrypted_msg)
        msg_bytes = zlib.decompress(decrypted_msg)
        msg = pickle.loads(msg_bytes)
        comp_time = time.time() - start_comp
        
        metrics = {
            'sent_bytes': 0,
            'received_bytes': msglen + 4,  # Include length header
            'comp_time': comp_time,
            'net_time': net_time,
            'original_size': len(msg_bytes),
            'compressed_size': msglen
        }
        
        return msg, metrics
        
    except Exception as e:
        logger.error(f"Error in secure_recv_msg: {e}")
        raise

def perform_handshake(sock, is_server=True):
    try:
        secure_conn = SecureConnection()
        
        if is_server:
            # Server generates and sends a challenge
            challenge = os.urandom(32)
            sock.sendall(struct.pack(">I", len(challenge)))
            sock.sendall(challenge)
            
            # Receive response from client
            response_len = struct.unpack(">I", recvall(sock, 4))[0]
            response = recvall(sock, response_len)
            
        else:
            # Client receives challenge
            challenge_len = struct.unpack(">I", recvall(sock, 4))[0]
            challenge = recvall(sock, challenge_len)
            
            # Client generates and sends response
            response = os.urandom(32)
            sock.sendall(struct.pack(">I", len(response)))
            sock.sendall(response)
        
        # Create shared secret
        shared_secret = challenge + response
        secure_conn.create_key(shared_secret)
        
        # Verify connection
        if is_server:
            verify_msg = b"SERVER_VERIFY"
            sock.sendall(struct.pack(">I", len(verify_msg)))
            sock.sendall(verify_msg)
            
            client_verify = recvall(sock, struct.unpack(">I", recvall(sock, 4))[0])
            if client_verify != b"CLIENT_VERIFY":
                raise Exception("Handshake verification failed")
        else:
            server_verify = recvall(sock, struct.unpack(">I", recvall(sock, 4))[0])
            if server_verify != b"SERVER_VERIFY":
                raise Exception("Handshake verification failed")
                
            verify_msg = b"CLIENT_VERIFY"
            sock.sendall(struct.pack(">I", len(verify_msg)))
            sock.sendall(verify_msg)
            
        return secure_conn
        
    except Exception as e:
        logger.error(f"Error in handshake: {e}")
        raise

def recvall(sock, n):
    """Optimized data receiving."""
    # Pre-allocate buffer for better performance
    data = bytearray(n)
    view = memoryview(data)
    pos = 0
    
    while pos < n:
        # Use the pre-allocated buffer
        received = sock.recv_into(view[pos:])
        if not received:
            return None
        pos += received
    
    return bytes(data)

def configure_socket(sock):
    """Configure socket for optimal performance."""
    # Enable TCP_NODELAY for faster small messages
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    # Increase buffer sizes
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
    
    # Enable TCP keepalive
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

class CentralServer:
    def __init__(self, model, dataset, batch_size=16, ip="localhost", port=5555,
                 checkpoint_dir="checkpoints", checkpoint_interval=5, secure=False):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.ip = ip
        self.port = port
        self.nodes = []
        self.batch_queue = queue.Queue(maxsize=1000)
        self.gradient_queue = queue.Queue()
        self.running = True
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint = time.time()
        self.global_step = 0
        self.secure = secure

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
        logger.info(f"New node connected: {addr}")
        try:
            if self.secure:
                secure_conn = perform_handshake(conn, is_server=True)
                logger.info(f"Completed handshake with node {addr}")
            
            while self.running:
                if not self.batch_queue.empty():
                    batch = self.batch_queue.get()
                    logger.info(f"Sending batch {batch['batch_id']} to node {addr}")
                    
                    # Send batch without collecting metrics on server side
                    if self.secure:
                        secure_send_msg(conn, batch, secure_conn, False)
                    else:
                        send_msg(conn, batch, False)
                    
                    logger.info(f"Waiting for gradients from node {addr}")
                    
                    # Receive gradients without collecting metrics
                    if self.secure:
                        gradients_data = secure_recv_msg(conn, secure_conn, False)
                    else:
                        gradients_data = recv_msg(conn, False)
                        
                    if gradients_data is None:
                        logger.warning(f"Received None gradients from {addr}")
                        break
                    
                    # Convert received gradients back to tensors
                    gradients = [
                        torch.tensor(grad) if grad is not None else None 
                        for grad in gradients_data['gradients']
                    ]
                    
                    logger.info(f"Received gradients for batch {gradients_data['batch_id']}")
                    self.gradient_queue.put(gradients)
                else:
                    logger.debug("Batch queue is empty, waiting...")
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error handling node {addr}: {e}")
        finally:
            logger.info(f"Node {addr} disconnected")
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
    def __init__(self, model, server_address=('localhost', 5555), secure=False, collect_metrics=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.server_address = server_address
        self.optimizer = torch.optim.Adam(model.parameters())
        self.running = True
        self.secure = secure
        self.collect_metrics = collect_metrics
        logger.info(f"Using device: {self.device}")
    
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
        sock.connect(self.server_address)
        
        try:
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
                
                # Send gradients
                gradients_data = {
                    'batch_id': batch['batch_id'],
                    'gradients': [
                        grad.cpu().numpy().tolist() if grad is not None else None 
                        for grad in [param.grad for param in self.model.parameters()]
                    ]
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