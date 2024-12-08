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

def send_msg(sock, msg):
    msg = pickle.dumps(msg)
    msg = zlib.compress(msg)
    msg_len = len(msg)
    sock.sendall(struct.pack(">I", msg_len))
    sock.sendall(msg)

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    msg = recvall(sock, msglen)
    if msg is None:
        return None
    msg = zlib.decompress(msg)
    return pickle.loads(msg)

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

def secure_send_msg(sock, msg, secure_conn):
    try:
        pickled_msg = pickle.dumps(msg)
        compressed_msg = zlib.compress(pickled_msg)
        encrypted_msg = secure_conn.encrypt_message(compressed_msg)
        msg_len = len(encrypted_msg)
        sock.sendall(struct.pack(">I", msg_len))
        sock.sendall(encrypted_msg)
        
    except Exception as e:
        logger.error(f"Error in secure_send_msg: {e}")
        raise

def secure_recv_msg(sock, secure_conn):
    try:
        raw_msglen = recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        encrypted_msg = recvall(sock, msglen)
        if encrypted_msg is None:
            return None
        decrypted_msg = secure_conn.decrypt_message(encrypted_msg)
        decompressed_msg = zlib.decompress(decrypted_msg)
        return pickle.loads(decompressed_msg)
        
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
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)

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
                    
                    if self.secure:
                        secure_send_msg(conn, batch, secure_conn)
                    else:
                        send_msg(conn, batch)
                    
                    logger.info(f"Waiting for gradients from node {addr}")
                    if self.secure:
                        gradients_data = secure_recv_msg(conn, secure_conn)
                    else:
                        gradients_data = recv_msg(conn)
                        
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
    def __init__(self, model, server_address=('localhost', 5555), secure=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.server_address = server_address
        self.optimizer = torch.optim.Adam(model.parameters())
        self.running = True
        self.secure = secure
        logger.info(f"Using device: {self.device}")
    
    def train(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info(f"Connecting to server at {self.server_address}")
        sock.connect(self.server_address)
        
        try:
            if self.secure:
                secure_conn = perform_handshake(sock, is_server=False)
                logger.info("Completed handshake with server")
            
            while self.running:
                logger.info("Waiting for batch from server...")
                if self.secure:
                    batch = secure_recv_msg(sock, secure_conn)
                else:
                    batch = recv_msg(sock)
                if batch is None:
                    logger.warning("Received None batch")
                    break

                logger.info(f"Received batch {batch['batch_id']}")

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device) if batch.get('attention_mask') is not None else None
                labels = batch['labels'].to(self.device) if batch.get('labels') is not None else None

                self.optimizer.zero_grad()

                outputs = self.model(input_ids)
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, outputs.size(-1)), 
                    input_ids.view(-1)
                )
                
                loss.backward()
                
                # Convert gradients to a serializable format
                gradients_data = {
                    'batch_id': batch['batch_id'],
                    'gradients': [
                        grad.cpu().numpy().tolist() if grad is not None else None 
                        for grad in [param.grad for param in self.model.parameters()]
                    ]
                }
                
                logger.info(f"Sending gradients for batch {batch['batch_id']}")
                if self.secure:
                    secure_send_msg(sock, gradients_data, secure_conn)
                else:
                    send_msg(sock, gradients_data)
                
                logger.info(f"Training loss: {loss.item():.4f}")
                
        except Exception as e:
            logger.error(f"Error in training node: {e}", exc_info=True)
        finally:
            logger.info("Closing connection")
            sock.close()