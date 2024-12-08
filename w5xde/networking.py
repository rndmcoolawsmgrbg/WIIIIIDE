import base64
import errno
import time
import os
import pickle
import struct
import lz4.block
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
import socket
import select
import sys
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TORCH_TO_NUMPY_DTYPE = {
    'torch.float32': 'float32',
    'torch.float64': 'float64',
    'torch.float16': 'float16',
    'torch.int64': 'int64',
    'torch.int32': 'int32',
    'torch.int16': 'int16',
    'torch.uint8': 'uint8',
    'torch.int8': 'int8',
    'torch.bool': 'bool',
    'torch.complex64': 'complex64',
    'torch.complex128': 'complex128',
    'torch.bfloat16': 'float32',  # bfloat16 doesn't exist in numpy, convert to float32
}

def send_msg(sock, msg, collect_metrics=False):
    """Adaptive message sending with dynamic chunking."""
    if collect_metrics:
        start_comp = time.time()
    
    # Adaptive serialization
    if isinstance(msg, dict):
        serialized_data = {}
        for key, value in msg.items():
            if isinstance(value, torch.Tensor):
                # Get tensor size and adapt chunk size accordingly
                tensor_bytes = value.numel() * value.element_size()
                # Use smaller chunks for larger tensors
                chunk_size = max(1024, min(tensor_bytes // 10, 1024 * 1024))
                
                # Detach and move to CPU
                value = value.detach().cpu()
                if value.dtype == torch.bfloat16:
                    value = value.float()
                
                # Split large tensors
                if tensor_bytes > chunk_size:
                    num_chunks = (tensor_bytes + chunk_size - 1) // chunk_size
                    splits = value.split(value.shape[0] // num_chunks)
                    serialized_data[key] = {
                        'type': 'tensor_chunks',
                        'chunks': [chunk.numpy() for chunk in splits],
                        'shape': value.shape,
                        'dtype': str(value.dtype)
                    }
                else:
                    serialized_data[key] = {
                        'type': 'tensor',
                        'data': value.numpy(),
                        'shape': value.shape,
                        'dtype': str(value.dtype)
                    }
            else:
                serialized_data[key] = {
                    'type': 'pickle',
                    'data': pickle.dumps(value, protocol=5)
                }
        
        msg_bytes = pickle.dumps(serialized_data, protocol=5)
    else:
        msg_bytes = pickle.dumps(msg, protocol=5)
    
    # Adaptive compression based on data size
    if len(msg_bytes) > 1024 * 1024:  # 1MB
        compressed = lz4.block.compress(msg_bytes, mode='fast', acceleration=1)
    else:
        compressed = msg_bytes
    
    is_compressed = len(msg_bytes) > 1024 * 1024
    
    if collect_metrics:
        comp_time = time.time() - start_comp
        start_net = time.time()
    
    try:
        # Send metadata
        metadata = {
            'size': len(compressed),
            'compressed': is_compressed
        }
        metadata_bytes = pickle.dumps(metadata, protocol=5)
        header = struct.pack(">I", len(metadata_bytes))
        sock.sendall(header)
        sock.sendall(metadata_bytes)
        
        # Send data in adaptive chunks
        chunk_size = min(8192, len(compressed))  # Start small
        pos = 0
        while pos < len(compressed):
            # Increase chunk size gradually if transfer is smooth
            if pos > 0 and pos % (chunk_size * 10) == 0:
                chunk_size = min(chunk_size * 2, 1024 * 1024)
            
            end = min(pos + chunk_size, len(compressed))
            sock.sendall(compressed[pos:end])
            pos = end
            
    except Exception as e:
        logger.error(f"Send error: {e}")
        raise
    
    if collect_metrics:
        net_time = time.time() - start_net
        return {
            'sent_bytes': len(compressed),
            'received_bytes': 0,
            'comp_time': comp_time,
            'net_time': net_time,
            'original_size': len(msg_bytes),
            'compressed_size': len(compressed)
        }
    return None

def recv_msg(sock, collect_metrics=False):
    """Adaptive message receiving with dynamic buffering."""
    if collect_metrics:
        start_net = time.time()
    
    try:
        # Receive metadata
        header = recvall(sock, 4)
        if header is None:
            return (None, None) if collect_metrics else None
            
        header_size = struct.unpack(">I", header)[0]
        metadata_bytes = recvall(sock, header_size)
        if metadata_bytes is None:
            return (None, None) if collect_metrics else None
            
        metadata = pickle.loads(metadata_bytes)
        
        # Receive data with adaptive buffering
        data = bytearray(metadata['size'])
        view = memoryview(data)
        pos = 0
        chunk_size = min(8192, metadata['size'])  # Start small
        
        while pos < metadata['size']:
            try:
                remaining = metadata['size'] - pos
                current_chunk = min(chunk_size, remaining)
                received = sock.recv_into(view[pos:pos + current_chunk], current_chunk)
                
                if not received:
                    return (None, None) if collect_metrics else None
                pos += received
                
                # Increase chunk size gradually if transfer is smooth
                if pos > 0 and pos % (chunk_size * 10) == 0:
                    chunk_size = min(chunk_size * 2, 1024 * 1024)
                    
            except socket.error as e:
                logger.error(f"Socket error during receive: {e}")
                return (None, None) if collect_metrics else None
        
        if collect_metrics:
            net_time = time.time() - start_net
            start_comp = time.time()
        
        try:
            # Decompress if needed
            if metadata['compressed']:
                msg_bytes = lz4.block.decompress(data)
            else:
                msg_bytes = data
            
            # Deserialize
            received_data = pickle.loads(msg_bytes)
            
            if isinstance(received_data, dict):
                deserialized = {}
                for key, value in received_data.items():
                    if value['type'] == 'tensor_chunks':
                        # Reconstruct tensor from chunks
                        chunks = [torch.from_numpy(chunk) for chunk in value['chunks']]
                        deserialized[key] = torch.cat(chunks).reshape(value['shape'])
                    elif value['type'] == 'tensor':
                        deserialized[key] = torch.from_numpy(value['data'])
                    else:  # 'pickle'
                        deserialized[key] = pickle.loads(value['data'])
                received_data = deserialized
            
            if collect_metrics:
                comp_time = time.time() - start_comp
                metrics = {
                    'sent_bytes': 0,
                    'received_bytes': metadata['size'],
                    'comp_time': comp_time,
                    'net_time': net_time,
                    'original_size': len(msg_bytes),
                    'compressed_size': metadata['size']
                }
                return received_data, metrics
            
            return received_data
            
        except Exception as e:
            logger.error(f"Error processing received data: {e}")
            return (None, None) if collect_metrics else None
            
    except Exception as e:
        logger.error(f"Receive error: {e}")
        return (None, None) if collect_metrics else None

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
        # Ultra-fast compression for secure messages
        compressed_msg = lz4.block.compress(
            pickled_msg,
            mode='fast',
            acceleration=8,  # Maximum speed
            store_size=False
        )
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
                'sent_bytes': msg_len + 4,
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
        msg_bytes = lz4.block.decompress(decrypted_msg)
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
    """Basic receive all bytes."""
    data = bytearray(n)
    pos = 0
    while pos < n:
        received = sock.recv_into(memoryview(data)[pos:], n - pos)
        if not received:
            return None
        pos += received
    return data

def configure_socket(sock):
    """Enhanced socket configuration."""
    try:
        # Use smaller buffer sizes
        BUFFER_SIZE = 1024 * 1024  # 1MB buffer
        
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
        
        # Enable TCP optimizations
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        
        # Don't set a global timeout - handle timeouts per operation
        sock.setblocking(True)
        
    except Exception as e:
        logger.warning(f"Socket configuration warning: {e}")

def configure_server_socket(server_socket):
    """Configure server-specific socket settings."""
    try:
        # Make server socket non-blocking
        server_socket.setblocking(False)
        
        # Use smaller buffer sizes
        BUFFER_SIZE = 1024 * 1024  # 1MB buffer
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, BUFFER_SIZE)
        
        # Enable address reuse
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    except Exception as e:
        logger.warning(f"Server socket configuration warning: {e}")

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
    
    sock.setblocking(True)