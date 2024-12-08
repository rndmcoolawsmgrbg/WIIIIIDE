import torch
import threading
import time
from simple_model import SimpleModel, SyntheticDataset
from w5xde import CentralServer, TrainingNode
import logging
from datetime import datetime
import queue

# Set logging level to see more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def format_size(bytes):
    """Convert bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024
    return f"{bytes:.2f}TB"

def format_throughput(bytes_per_sec):
    """Convert bytes/sec to human readable string."""
    return f"{format_size(bytes_per_sec)}/s"

def get_process_memory():
    """Get current process memory in bytes using vanilla Python."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Convert KB to bytes

class NetworkStats:
    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.compression_time = 0
        self.network_time = 0
        self.original_size = 0
        self.compressed_size = 0
        self.start_time = time.time()

    def update(self, sent, received, comp_time, net_time, orig_size, comp_size):
        self.bytes_sent += sent
        self.bytes_received += received
        self.compression_time += comp_time
        self.network_time += net_time
        self.original_size += orig_size
        self.compressed_size += comp_size

    def get_throughput(self):
        elapsed = time.time() - self.start_time
        total_bytes = self.bytes_sent + self.bytes_received
        return total_bytes / elapsed if elapsed > 0 else 0

    def get_compression_ratio(self):
        return (self.original_size / self.compressed_size 
                if self.compressed_size > 0 else 0)

    def print_summary(self):
        logger.info("\nNetwork Statistics:")
        logger.info("-" * 20)
        logger.info(f"Total data sent: {format_size(self.bytes_sent)}")
        logger.info(f"Total data received: {format_size(self.bytes_received)}")
        logger.info(f"Network throughput: {format_throughput(self.get_throughput())}")
        logger.info(f"Time spent in network ops: {self.network_time:.2f}s")
        
        logger.info("\nCompression Statistics:")
        logger.info("-" * 20)
        logger.info(f"Original data size: {format_size(self.original_size)}")
        logger.info(f"Compressed data size: {format_size(self.compressed_size)}")
        logger.info(f"Compression ratio: {self.get_compression_ratio():.2f}x")
        logger.info(f"Time spent compressing: {self.compression_time:.2f}s")

class TrainingStats:
    def __init__(self, num_nodes):
        self.start_time = None
        self.end_time = None
        self.total_batches = 0
        self.total_loss = 0.0
        self.losses = []
        self.loss_queue = queue.Queue()
        self.epochs_completed = 0
        self.num_nodes = num_nodes
        # Track per-node statistics
        self.node_stats = {i: {'batches': 0, 'total_loss': 0.0, 'losses': []} 
                          for i in range(num_nodes)}
        self.network_stats = {i: NetworkStats() for i in range(num_nodes)}
        
    def start(self):
        self.start_time = datetime.now()

    def stop(self):
        self.end_time = datetime.now()

    def add_loss(self, loss, batch_id, node_id):
        self.total_batches += 1
        self.total_loss += loss
        self.losses.append(loss)
        
        # Update per-node statistics
        self.node_stats[node_id]['batches'] += 1
        self.node_stats[node_id]['total_loss'] += loss
        self.node_stats[node_id]['losses'].append(loss)
        
        if "batch0" in batch_id and len(self.losses) > 1:
            self.epochs_completed += 1

    def get_training_time(self):
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def get_average_loss(self):
        return self.total_loss / self.total_batches if self.total_batches > 0 else 0

    def add_network_stats(self, node_id, sent, received, comp_time, net_time, orig_size, comp_size):
        self.network_stats[node_id].update(sent, received, comp_time, net_time, orig_size, comp_size)

    def print_summary(self):
        logger.info("=" * 50)
        logger.info("Training Summary:")
        logger.info("-" * 50)
        logger.info(f"Number of training nodes: {self.num_nodes}")
        logger.info(f"Total training time: {self.get_training_time():.2f} seconds")
        logger.info(f"Total batches processed: {self.total_batches}")
        logger.info(f"Batches per node: {self.total_batches / self.num_nodes:.1f}")
        logger.info(f"Epochs completed: {self.epochs_completed}")
        logger.info(f"Average loss: {self.get_average_loss():.4f}")
        if self.losses:
            logger.info(f"Initial loss: {self.losses[0]:.4f}")
            logger.info(f"Final loss: {self.losses[-1]:.4f}")
        
        # Per-node statistics
        logger.info("\nPer-node Statistics:")
        logger.info("-" * 20)
        for node_id, node_stat in self.node_stats.items():
            avg_loss = (node_stat['total_loss'] / node_stat['batches'] 
                       if node_stat['batches'] > 0 else 0)
            logger.info(f"Node {node_id}:")
            logger.info(f"  Batches processed: {node_stat['batches']}")
            logger.info(f"  Average loss: {avg_loss:.4f}")
            if node_stat['losses']:
                logger.info(f"  Initial loss: {node_stat['losses'][0]:.4f}")
                logger.info(f"  Final loss: {node_stat['losses'][-1]:.4f}")
        
        # Add network statistics per node
        logger.info("\nNetwork Statistics by Node:")
        logger.info("-" * 20)
        total_throughput = 0
        for node_id, net_stats in self.network_stats.items():
            throughput = net_stats.get_throughput()
            total_throughput += throughput
            logger.info(f"\nNode {node_id}:")
            logger.info(f"  Network throughput: {format_throughput(throughput)}")
            logger.info(f"  Compression ratio: {net_stats.get_compression_ratio():.2f}x")
            logger.info(f"  Network time: {net_stats.network_time:.2f}s")
            logger.info(f"  Compression time: {net_stats.compression_time:.2f}s")
        
        logger.info(f"\nTotal network throughput: {format_throughput(total_throughput)}")
        
        # System resource usage
        logger.info("\nSystem Resource Usage:")
        logger.info("-" * 20)
        logger.info(f"Memory Usage: {format_size(get_process_memory())}")
        logger.info("=" * 50)

def start_node(model, node_id):
    def loss_callback(loss, batch_id):
        stats.add_loss(loss, batch_id, node_id)
        
    def network_callback(sent, received, comp_time, net_time, orig_size, comp_size):
        stats.add_network_stats(node_id, sent, received, comp_time, net_time, orig_size, comp_size)
        
    node = TrainingNode(
        model=model,
        server_address=('localhost', 5555),
        secure=False,
        collect_metrics=True
    )
    node.train(loss_callback, network_callback)

def get_user_input(prompt):
    while True:
        try:
            response = input(prompt).lower().strip()
            if response in ['y', 'n']:
                return response == 'y'
            print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            return False

def run_training_session(num_nodes):
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        logger.info(f"Initializing training setup with {num_nodes} nodes...")
        
        # Create a simple model
        input_size = 10
        model = SimpleModel(input_size=input_size)
        
        # Create synthetic dataset
        dataset = SyntheticDataset(size=1000, input_size=input_size)
        
        # Initialize server
        server = CentralServer(
            model=model,
            dataset=dataset,
            batch_size=32,
            ip="localhost",
            port=5555,
            secure=False,
            checkpoint_interval=100
        )
        
        logger.info("Starting server...")
        server_thread = threading.Thread(target=server.start)
        server_thread.start()
        
        time.sleep(2)
        
        # Initialize global stats
        global stats
        stats = TrainingStats(num_nodes)
        
        # Create and start multiple training nodes
        node_threads = []
        logger.info(f"Starting {num_nodes} training nodes...")
        for i in range(num_nodes):
            node_model = SimpleModel(input_size=input_size)
            thread = threading.Thread(
                target=start_node, 
                args=(node_model, i)
            )
            node_threads.append(thread)
        
        stats.start()
        for thread in node_threads:
            thread.start()
        
        try:
            training_duration = 30
            logger.info(f"Training for {training_duration} seconds...")
            time.sleep(training_duration)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user...")
        finally:
            stats.stop()
            
            logger.info("Shutting down...")
            server.running = False
            
            for thread in node_threads:
                thread.join(timeout=2)
            server_thread.join(timeout=2)
            
            stats.print_summary()
            
            logger.info("Training completed successfully!")
            return True

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return False

def main():
    try:
        while True:
            # Get number of nodes
            try:
                num_nodes = int(input("Enter number of training nodes (default=2): ") or "2")
                if num_nodes < 1:
                    raise ValueError
            except ValueError:
                logger.error("Please enter a valid number of nodes (>= 1)")
                continue

            # Run training session
            success = run_training_session(num_nodes)
            
            # Ask to run again
            logger.info("\nWould you like to run another training session?")
            if not get_user_input("Run again? (y/n): "):
                logger.info("Exiting...")
                return 0  # Exit cleanly when user enters 'n'

    except KeyboardInterrupt:
        logger.info("\nExiting...")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)  # This will properly terminate the process 