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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_msg(sock, msg, collect_metrics=False):
    """Highly optimized message sending."""
    if collect_metrics:
        start_comp = time.time()
    
    # Use pickle protocol 5 with out-of-band buffer optimization
    msg_bytes = pickle.dumps(msg, protocol=5, buffer_callback=lambda x: x)
    original_size = len(msg_bytes) if collect_metrics else 0
    
    # Use different compression strategies based on data type and size
    if isinstance(msg, dict) and 'gradients' in msg:
        if len(msg_bytes) > 1024 * 1024:  # 1MB
            # For large gradients, use fast compression
            compressed_msg = zlib.compress(msg_bytes, level=1)
            is_compressed = True
        else:
            # For smaller gradients, skip compression
            compressed_msg = msg_bytes
            is_compressed = False
    else:
        compressed_msg = zlib.compress(msg_bytes, level=9)
        is_compressed = True
    
    compressed_size = len(compressed_msg) if collect_metrics else 0
    
    if collect_metrics:
        comp_time = time.time() - start_comp
        start_net = time.time()
    
    # Send compression flag first
    flag = struct.pack("?", is_compressed)
    msg_len = len(compressed_msg)
    header = struct.pack(">I", msg_len)
    
    try:
        # Send flag, length and data
        sock.sendall(flag + header + compressed_msg)
    except BlockingIOError:
        sock.sendall(flag)
        sock.sendall(header)
        sock.sendall(compressed_msg)
    
    if collect_metrics:
        net_time = time.time() - start_net
        return {
            'sent_bytes': msg_len + 5,  # +5 for header and flag
            'received_bytes': 0,
            'comp_time': comp_time,
            'net_time': net_time,
            'original_size': original_size,
            'compressed_size': compressed_size
        }
    return None

def recv_msg(sock, collect_metrics=False):
    """High-performance data receiving."""
    if collect_metrics:
        start_net = time.time()
    
    # Receive compression flag
    flag_bytes = recvall(sock, 1)
    if not flag_bytes:
        return (None, None) if collect_metrics else None
    is_compressed = struct.unpack("?", flag_bytes)[0]
    
    # Receive length
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return (None, None) if collect_metrics else None
    
    msglen = struct.unpack('>I', raw_msglen)[0]
    
    # Receive data
    msg_data = recvall(sock, msglen)
    if msg_data is None:
        return (None, None) if collect_metrics else None
    
    if collect_metrics:
        net_time = time.time() - start_net
        start_comp = time.time()
    
    # Decompress if necessary
    if is_compressed:
        msg_bytes = zlib.decompress(msg_data)
    else:
        msg_bytes = msg_data
    
    msg = pickle.loads(msg_bytes)
    
    if collect_metrics:
        comp_time = time.time() - start_comp
        metrics = {
            'sent_bytes': 0,
            'received_bytes': msglen + 5,  # +5 for header and flag
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
    """High-performance data receiving."""
    data = bytearray(n)
    view = memoryview(data)
    pos = 0
    
    # Use 256KB chunks for better throughput
    chunk_size = min(256 * 1024, n)
    
    while pos < n:
        try:
            while True:
                try:
                    received = sock.recv_into(view[pos:pos + chunk_size])
                    if not received:
                        return None
                    pos += received
                    break
                except BlockingIOError:
                    select.select([sock], [], [])
                    continue
                except socket.error as e:
                    if e.errno != errno.EAGAIN:
                        raise
                    select.select([sock], [], [])
        except Exception as e:
            logger.error(f"Receive error: {e}")
            raise
    
    return bytes(data)

def configure_socket(sock):
    """Configure socket for maximum throughput."""
    # Increase buffer sizes to 8MB for higher throughput
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    
    # Enable TCP_NODELAY and TCP_QUICKACK
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
    
    # Set TCP_FASTOPEN for faster connection establishment
    sock.setsockopt(socket.SOL_TCP, socket.TCP_FASTOPEN, 1)
    
    # Enable TCP window scaling
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

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

def connect_with_retry(sock, server_address, timeout=30):
    """Connect with retry for non-blocking sockets."""
    sock.setblocking(False)
    try:
        sock.connect(server_address)
    except BlockingIOError:
        pass
    
    start_time = time.time()
    while True:
        try:
            sock.getpeername()
            break  # Connected successfully
        except socket.error:
            if time.time() - start_time > timeout:
                raise TimeoutError("Connection timeout")
            # Wait for socket to be writable (connected)
            _, writable, _ = select.select([], [sock], [], 0.1)
            if writable:
                err = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if err != 0:
                    raise socket.error(err)
                break
    
    # Set back to blocking mode after connection
    sock.setblocking(True)

class CentralServer:
    def __init__(self, model, dataset, batch_size=16, ip="localhost", port=5555,
                 checkpoint_dir="checkpoints", checkpoint_interval=5, secure=False):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.ip = ip
        self.port = port
        self.nodes = []
        self.batch_queue = BatchQueue(maxsize=1000)  # Use optimized queue
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

class GradientCompression:
    def __init__(self, bits=8, scale_method='dynamic'):
        self.bits = bits
        self.scale_method = scale_method
        self.max_val = 2**(bits-1) - 1
        self.min_val = -(2**(bits-1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute constants
        self.scale_factor = float(self.max_val)
        
        # Reusable buffers
        self.buffers = {}
    
    def _get_buffer(self, shape, grad_id):
        key = (grad_id, shape)
        if key not in self.buffers:
            self.buffers[key] = {
                'error': torch.zeros(shape, dtype=torch.float32, device=self.device),
                'quantized': torch.zeros(shape, dtype=torch.int8, device=self.device),
                'temp': torch.zeros(shape, dtype=torch.float32, device=self.device)
            }
        return self.buffers[key]
    
    def compress(self, gradients, grad_id=None):
        """Fast compression with minimal memory operations."""
        compressed_grads = []
        scales = []
        
        for i, grad in enumerate(gradients):
            if grad is None:
                compressed_grads.append(None)
                scales.append(None)
                continue
            
            buffer = self._get_buffer(grad.shape, f"{grad_id}_{i}")
            
            with torch.no_grad():
                # Move to device and add error in single operation
                torch.add(grad.to(self.device), buffer['error'], out=buffer['temp'])
                
                # Fast scale calculation
                if self.scale_method == 'dynamic':
                    scale = buffer['temp'].abs().max().clamp_(min=1e-8)
                else:
                    scale = 1.0
                
                # Quantize in-place
                buffer['temp'].div_(scale).mul_(self.scale_factor)
                buffer['temp'].round_().clamp_(self.min_val, self.max_val)
                buffer['quantized'].copy_(buffer['temp'])
                
                # Update error feedback
                buffer['temp'].mul_(scale / self.scale_factor)
                torch.sub(grad.to(self.device), buffer['temp'], out=buffer['error'])
                
                # Apply fast compression to quantized data
                quantized_data = buffer['quantized'].cpu()
                compressed_data = self.compressor.compress(quantized_data.numpy().tobytes())
                
                # Store compressed form
                compressed_grads.append(compressed_data)
                scales.append(scale.item())
        
        return compressed_grads, scales
    
    def decompress(self, compressed_grads, scales):
        """Fast decompression."""
        gradients = []
        
        for grad_bytes, scale in zip(compressed_grads, scales):
            if grad_bytes is None:
                gradients.append(None)
                continue
            
            with torch.no_grad():
                # Decompress bytes back to tensor
                decompressed_bytes = self.compressor.decompress(grad_bytes)
                grad = torch.frombuffer(decompressed_bytes, dtype=torch.int8)
                
                # Convert to float and scale
                dequantized = grad.to(self.device).float().mul_(scale / self.scale_factor)
                gradients.append(dequantized)
        
        return gradients

class TrainingNode:
    def __init__(self, model, server_address=('localhost', 5555), 
                 secure=False, collect_metrics=False, compress_gradients=False,
                 batch_gradients=True):
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

class FastCompression:
    """Fast compression optimized for gradient data using LZ4-like algorithm."""
    
    def __init__(self, window_size=64*1024):
        self.window_size = window_size
        self.hash_table = {}
    
    def _hash(self, data, pos):
        """4-byte rolling hash."""
        if pos + 4 > len(data):
            return 0
        return (data[pos] | 
                (data[pos + 1] << 8) | 
                (data[pos + 2] << 16) | 
                (data[pos + 3] << 24))
    
    def compress(self, data: bytes) -> bytes:
        """Fast compression optimized for numerical data."""
        if len(data) < 4:
            return data
        
        compressed = bytearray()
        pos = 0
        self.hash_table.clear()
        
        while pos < len(data):
            # Look for matches
            best_len = 3  # Minimum match length
            best_offset = 0
            
            if pos + best_len <= len(data):
                hash_val = self._hash(data, pos)
                if hash_val in self.hash_table:
                    prev_pos = self.hash_table[hash_val]
                    offset = pos - prev_pos
                    
                    if offset <= self.window_size:
                        # Find match length
                        match_len = 0
                        while (pos + match_len < len(data) and 
                               prev_pos + match_len < pos and 
                               data[pos + match_len] == data[prev_pos + match_len] and
                               match_len < 255):
                            match_len += 1
                        
                        if match_len >= best_len:
                            best_len = match_len
                            best_offset = offset
            
            # Store position in hash table
            if pos + 4 <= len(data):
                self.hash_table[hash_val] = pos
            
            # Output token
            if best_offset:
                # Match found
                token = (0x80 | (best_len - 3))  # Set high bit for match
                compressed.append(token)
                # Store offset in 2 bytes, little endian
                compressed.extend(best_offset.to_bytes(2, 'little'))
                pos += best_len
            else:
                # Literal
                compressed.append(data[pos])
                pos += 1
        
        return bytes(compressed)
    
    def decompress(self, data: bytes) -> bytes:
        """Fast decompression."""
        if len(data) < 4:
            return data
        
        decompressed = bytearray()
        pos = 0
        
        while pos < len(data):
            token = data[pos]
            pos += 1
            
            if token & 0x80:  # Match
                length = (token & 0x7F) + 3
                offset = int.from_bytes(data[pos:pos+2], 'little')
                pos += 2
                
                start = len(decompressed) - offset
                for i in range(length):
                    decompressed.append(decompressed[start + i])
            else:  # Literal
                decompressed.append(token)
        
        return bytes(decompressed)