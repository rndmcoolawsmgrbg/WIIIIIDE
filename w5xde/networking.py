import base64
import errno
import time
import os
import pickle
import struct
import zlib
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
import socket
import select
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_msg(sock, msg, collect_metrics=False):
    """Highly optimized message sending."""
    if collect_metrics:
        start_comp = time.time()
    
    # Use pickle protocol 5 with out-of-band buffer optimization
    msg_bytes = pickle.dumps(msg, protocol=5, buffer_callback=lambda x: x)
    original_size = len(msg_bytes) if collect_metrics else 0
    
    # Optimize compression strategy
    if isinstance(msg, dict) and 'gradients' in msg:
        if len(msg_bytes) > 512 * 1024:  # Reduced threshold to 512KB
            # Use fastest compression for large gradients
            compressed_msg = zlib.compress(msg_bytes, level=1)
            is_compressed = True
        else:
            # Skip compression for smaller messages
            compressed_msg = msg_bytes
            is_compressed = False
    else:
        # Use moderate compression for non-gradient data
        compressed_msg = zlib.compress(msg_bytes, level=3)  # Reduced from level 9
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
        # Send in a single call if possible
        sock.sendall(flag + header + compressed_msg)
    except BlockingIOError:
        # Fall back to chunked sending if needed
        sock.sendall(flag)
        sock.sendall(header)
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        for i in range(0, len(compressed_msg), CHUNK_SIZE):
            chunk = compressed_msg[i:i + CHUNK_SIZE]
            sock.sendall(chunk)
    
    if collect_metrics:
        net_time = time.time() - start_net
        return {
            'sent_bytes': msg_len + 5,
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
    """Zero-copy high-performance data receiving."""
    data = bytearray(n)
    view = memoryview(data)
    pos = 0
    
    # Increased chunk size for better throughput
    chunk_size = min(1024 * 1024, n)  # 1MB chunks
    
    while pos < n:
        try:
            # Direct recv_into with larger chunks
            received = sock.recv_into(view[pos:pos + chunk_size], chunk_size)
            if not received:
                return None
            pos += received
        except BlockingIOError:
            # Minimal polling with select
            select.select([sock], [], [], 0.001)  # 1ms timeout
            continue
        except socket.error as e:
            if e.errno != errno.EAGAIN:
                raise
            select.select([sock], [], [], 0.001)
    
    return bytes(data)

def configure_socket(sock):
    """Enhanced socket configuration for maximum performance."""
    try:
        # Increase buffer sizes significantly
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
        
        # Enable TCP optimizations
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        
        # Platform specific optimizations
        if sys.platform.startswith('linux'):
            # Linux-specific TCP optimizations
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                
                # TCP FastOpen (23)
                sock.setsockopt(socket.IPPROTO_TCP, 23, 1)
                
                # TCP thin-stream optimizations
                # TCP_THIN_LINEAR_TIMEOUTS (16)
                sock.setsockopt(socket.IPPROTO_TCP, 16, 1)
                # TCP_THIN_DUPACK (17)
                sock.setsockopt(socket.IPPROTO_TCP, 17, 1)
            except (AttributeError, OSError):
                pass
        
        elif sys.platform == 'darwin':
            try:
                # MacOS specific optimizations
                # TCP_NOTSENT_LOWAT (25)
                sock.setsockopt(socket.IPPROTO_TCP, 25, 16384)
            except (AttributeError, OSError):
                pass

    except OSError as e:
        logger.error(f"Socket configuration error: {e}, Ignoring")

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