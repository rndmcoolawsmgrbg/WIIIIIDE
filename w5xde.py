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

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


class CentralServer:
    def __init__(self, model, dataset, batch_size=16, ip="localhost", port=5555,
                 checkpoint_dir="checkpoints", checkpoint_interval=5):
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
            while self.running:
                if not self.batch_queue.empty():
                    batch = self.batch_queue.get()
                    logger.info(f"Sending batch {batch['batch_id']} to node {addr}")
                    send_msg(conn, batch)
                    
                    logger.info(f"Waiting for gradients from node {addr}")
                    gradients = recv_msg(conn)
                    if gradients is None:
                        logger.warning(f"Received None gradients from {addr}")
                        break
                    
                    logger.info(f"Received gradients from node {addr}")
                    self.gradient_queue.put(gradients)
                else:
                    logger.debug("Batch queue is empty, waiting...")
                    time.sleep(0.1)

                if self.global_step % self.checkpoint_interval == 0:
                    logger.info("Saving checkpoint...")
                    torch.save(self.model.state_dict(), f"{self.checkpoint_dir}/model_{self.global_step}.pt")
                    self.last_checkpoint = time.time()
                    logger.info("Checkpoint saved")
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
    def __init__(self, model, server_address=('localhost', 5555)):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.server_address = server_address
        self.optimizer = torch.optim.Adam(model.parameters())
        self.running = True
        logger.info(f"Using device: {self.device}")
    
    def train(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info(f"Connecting to server at {self.server_address}")
        sock.connect(self.server_address)
        
        try:
            while self.running:
                logger.info("Waiting for batch from server...")
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
                gradients = [param.grad.data.cpu() for param in self.model.parameters()]
                
                logger.info(f"Sending gradients for batch {batch['batch_id']}")
                send_msg(sock, gradients)
                
                logger.info(f"Training loss: {loss.item():.4f}")
                
        except Exception as e:
            logger.error(f"Error in training node: {e}", exc_info=True)
        finally:
            logger.info("Closing connection")
            sock.close()