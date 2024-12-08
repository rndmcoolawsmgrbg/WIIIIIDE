import torch
import threading
import time
from simple_model import SimpleModel, SyntheticDataset
from w5xde import CentralServer, TrainingNode
from training_viz import TrainingVisualizer
import logging
from datetime import datetime
import queue
import os
import socket
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
visualizer = None
stats = None
DEFAULT_TRAINING_DURATION = 30  # seconds
DEFAULT_PORT = 5555

class NetworkStats:
    """Tracks network-related statistics during training"""
    
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

class TrainingStats:
    """Tracks training-related statistics"""
    
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
        """Print comprehensive training summary"""
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
        
        # Network statistics
        logger.info("\nNetwork Statistics:")
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
        logger.info("=" * 50)

class TrainingManager:
    """Manages the distributed training process"""
    
    def __init__(self):
        self.running = True
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
    
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("\nShutdown signal received. Cleaning up...")
        self.running = False
        cleanup_resources()
        os._exit(0)
    
    def get_training_config(self):
        """Get training configuration from user input"""
        try:
            # Get number of nodes
            num_nodes = int(input("Enter number of training nodes (default=2): ") or "2")
            if num_nodes < 1:
                raise ValueError("Number of nodes must be >= 1")
            
            # Ask about gradient compression
            use_compression = get_user_input("Enable gradient compression? (y/n): ")
            
            # Ask about visualization
            use_visualization = get_user_input("Enable training visualization? (y/n): ")
            
            # Get logging mode
            print("\nLogging modes:")
            print("1. Silent (default) - only final metrics")
            print("2. Normal - basic progress updates")
            print("3. Verbose - detailed logging")
            log_mode = input("Select logging mode (1-3, default=1): ").strip() or "1"
            
            return num_nodes, use_compression, use_visualization, log_mode
            
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
            return None
    
    def configure_logging(self, log_mode):
        """Configure logging levels based on mode"""
        if log_mode == "1":  # Silent
            logging.getLogger('w5xde').setLevel(logging.ERROR)
            logging.getLogger('__main__').setLevel(logging.ERROR)
        elif log_mode == "2":  # Normal
            logging.getLogger('w5xde').setLevel(logging.WARNING)
            logging.getLogger('__main__').setLevel(logging.INFO)
        else:  # Verbose
            logging.getLogger('w5xde').setLevel(logging.INFO)
            logging.getLogger('__main__').setLevel(logging.DEBUG)
    
    def print_final_metrics(self):
        """Print final training metrics"""
        if not stats:
            return
        
        logger.info("\nFinal Training Metrics:")
        logger.info("-" * 20)
        logger.info(f"Total batches processed: {stats.total_batches}")
        logger.info(f"Total network throughput: {format_throughput(sum(ns.get_throughput() for ns in stats.network_stats.values()))}")
        logger.info(f"Average compression ratio: {sum(ns.get_compression_ratio() for ns in stats.network_stats.values()) / len(stats.network_stats):.2f}x")
        logger.info(f"Memory Usage: {format_size(get_process_memory())}")
    
    def run(self):
        """Main training loop"""
        global visualizer
        
        while self.running:
            # Get configuration
            config = self.get_training_config()
            if not config:
                continue
            
            num_nodes, use_compression, use_visualization, log_mode = config
            self.configure_logging(log_mode)
            
            # Initialize visualizer only if requested
            if use_visualization and visualizer is None:
                visualizer = TrainingVisualizer()
                visualizer.start()
            
            # Run training session
            success = run_training_session(num_nodes, use_compression)
            
            # Print metrics
            self.print_final_metrics()
            
            # Cleanup visualization
            if visualizer:
                visualizer.shutdown()
                visualizer = None
            
            # Ask to run again
            print("\nRun again? (y/n): ", end='', flush=True)
            response = input().lower().strip()
            if response != 'y':
                logger.info("Exiting...")
                cleanup_resources()
                break

def cleanup_resources():
    """Cleanup all resources"""
    cleanup_ports()
    if visualizer:
        visualizer.shutdown()

def get_user_input(prompt, options=None):
    """Enhanced user input handler with validation"""
    while True:
        try:
            response = input(prompt).lower().strip()
            if options:
                if response in options:
                    return response
                print(f"Please enter one of: {', '.join(options)}")
            else:
                if response in ['y', 'n']:
                    return response == 'y'
                print("Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            return False

def cleanup_ports():
    """Force cleanup of ports in use"""
    for port in [DEFAULT_PORT]:
        try:
            temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            temp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            temp_socket.bind(('localhost', port))
            temp_socket.close()
        except Exception as e:
            logger.warning(f"Could not cleanup port {port}: {e}")
            time.sleep(1)
            try:
                temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                temp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                temp_socket.bind(('localhost', port))
                temp_socket.close()
            except Exception as e:
                logger.error(f"Port {port} cleanup failed: {e}")

def format_size(bytes):
    """Convert bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f}{unit}"
        bytes /= 1024
    return f"{bytes:.2f}TB"

def format_throughput(bytes_per_sec):
    """Convert bytes/sec to human readable string"""
    return f"{format_size(bytes_per_sec)}/s"

def get_process_memory():
    """Get current process memory usage"""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

def run_training_session(num_nodes, use_compression):
    """Run a single training session"""
    global visualizer, stats
    
    try:
        # Initialize visualizer only if it exists (was previously created)
        if visualizer:
            visualizer.start()
        
        # Clean up any existing ports first
        cleanup_ports()
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        logger.info(f"Initializing training setup with {num_nodes} nodes...")
        logger.info(f"Gradient compression: {'enabled' if use_compression else 'disabled'}")
        
        # Create model and dataset
        input_size = 10
        model = SimpleModel(input_size=input_size)
        dataset = SyntheticDataset(size=1000, input_size=input_size)
        
        # Initialize server
        server = CentralServer(
            model=model,
            dataset=dataset,
            batch_size=32,
            ip="localhost",
            port=DEFAULT_PORT,
            secure=False,
            checkpoint_interval=100
        )
        
        # Start server
        logger.info("Starting server...")
        server_thread = threading.Thread(target=server.start)
        server_thread.start()
        time.sleep(2)  # Give server time to start
        
        # Initialize stats
        stats = TrainingStats(num_nodes)
        
        # Create and start training nodes
        node_threads = []
        logger.info(f"Starting {num_nodes} training nodes...")
        for i in range(num_nodes):
            node_model = SimpleModel(input_size=input_size)
            thread = threading.Thread(
                target=start_node,
                args=(node_model, i, use_compression)
            )
            node_threads.append(thread)
        
        # Start training
        stats.start()
        for thread in node_threads:
            thread.start()
        
        # Wait for training duration
        try:
            logger.info(f"Training for {DEFAULT_TRAINING_DURATION} seconds...")
            time.sleep(DEFAULT_TRAINING_DURATION)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user...")
        finally:
            stats.stop()
            
            logger.info("Shutting down...")
            server.running = False
            
            # Clean shutdown of threads
            for thread in node_threads:
                thread.join(timeout=2)
            server_thread.join(timeout=2)
            
            cleanup_ports()
            stats.print_summary()
            logger.info("Training completed successfully!")
            return True

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        cleanup_ports()
        if visualizer:
            visualizer.shutdown()
        return False

def start_node(model, node_id, use_compression):
    """Start a training node"""
    def loss_callback(loss, batch_id):
        stats.add_loss(loss, batch_id, node_id)
        # Add visualization update only if visualizer exists
        if visualizer:
            visualizer.update_data({
                'loss': loss,
                'node_id': node_id,
                'network_stats': {
                    node_id: {
                        'throughput': stats.network_stats[node_id].get_throughput() / 1024 / 1024,  # Convert to MB/s
                        'compression': stats.network_stats[node_id].get_compression_ratio()
                    }
                }
            })
    
    def network_callback(sent, received, comp_time, net_time, orig_size, comp_size):
        stats.add_network_stats(node_id, sent, received, comp_time, net_time, orig_size, comp_size)
    
    node = TrainingNode(
        model=model,
        server_address=('localhost', DEFAULT_PORT),
        secure=False,
        collect_metrics=True,
        compress_gradients=use_compression,
        batch_gradients=True
    )
    node.train(loss_callback, network_callback)

def main():
    """Main entry point"""
    try:
        manager = TrainingManager()
        manager.run()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        cleanup_resources()
        os._exit(1)

if __name__ == "__main__":
    main() 