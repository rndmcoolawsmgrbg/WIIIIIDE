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

class TrainingStats:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_batches = 0
        self.total_loss = 0.0
        self.losses = []
        self.loss_queue = queue.Queue()
        self.epochs_completed = 0

    def start(self):
        self.start_time = datetime.now()

    def stop(self):
        self.end_time = datetime.now()

    def add_loss(self, loss, batch_id):
        self.total_batches += 1
        self.total_loss += loss
        self.losses.append(loss)
        if "batch0" in batch_id and len(self.losses) > 1:
            self.epochs_completed += 1

    def get_training_time(self):
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

    def get_average_loss(self):
        return self.total_loss / self.total_batches if self.total_batches > 0 else 0

    def print_summary(self):
        logger.info("=" * 50)
        logger.info("Training Summary:")
        logger.info("-" * 50)
        logger.info(f"Total training time: {self.get_training_time():.2f} seconds")
        logger.info(f"Total batches processed: {self.total_batches}")
        logger.info(f"Epochs completed: {self.epochs_completed}")
        logger.info(f"Average loss: {self.get_average_loss():.4f}")
        if self.losses:
            logger.info(f"Initial loss: {self.losses[0]:.4f}")
            logger.info(f"Final loss: {self.losses[-1]:.4f}")
        logger.info("=" * 50)

stats = TrainingStats()

def start_node(model):
    def loss_callback(loss, batch_id):
        stats.add_loss(loss, batch_id)
        
    node = TrainingNode(
        model=model,
        server_address=('localhost', 5555),
        secure=False
    )
    node.train(loss_callback)

def main():
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        logger.info("Initializing training setup...")
        
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
        
        logger.info("Starting training node...")
        node_model = SimpleModel(input_size=input_size)
        node_thread = threading.Thread(target=start_node, args=(node_model,))
        
        stats.start()
        node_thread.start()
        
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
            
            node_thread.join(timeout=2)
            server_thread.join(timeout=2)
            
            stats.print_summary()
            
            logger.info("Training completed successfully!")
            return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 