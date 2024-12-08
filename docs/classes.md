# Classes, Arguments, and Methods

## Classes

W5XDE contains two main classes: `CentralServer` and `TrainingNode`.

### CentralServer

The `CentralServer` class is the central hub for distributed training, handling batch distribution and gradient aggregation.

#### Arguments

- `model` (required): PyTorch model to train
- `dataset` (required): PyTorch dataset for training
- `batch_size` (int): Size of training batches. Default: 16
- `ip` (str): IP address to bind server. Default: "localhost". Use "0.0.0.0" for all interfaces
- `port` (int): Port number. Default: 5555
- `checkpoint_dir` (str): Directory for model checkpoints. Default: "checkpoints"
- `checkpoint_interval` (int): Minutes between checkpoints. Default: 5
- `secure` (bool): Enable encrypted communication. Default: False
- `queue_size` (int): Size of batch queue. Default: 1000

#### Methods
- `start()`: Starts the server (blocking call)

### TrainingNode

The `TrainingNode` class connects to the `CentralServer` to receive batches and send gradients.

#### Arguments

- `model` (required): PyTorch model matching server's model
- `server_address` (tuple): Server (ip, port). Default: ('localhost', 5555)
- `secure` (bool): Enable encrypted communication. Default: False
- `collect_metrics` (bool): Enable performance metrics. Default: False
- `compress_gradients` (bool): Enable gradient compression. Default: False
- `batch_gradients` (bool): Batch gradients before sending. Default: True

#### Methods
- `train(loss_callback=None, network_callback=None)`: Start training
  - `loss_callback(loss_value: float, batch_id: str)`: Track training loss
  - `network_callback(sent_bytes: int, received_bytes: int, comp_time: float, net_time: float, orig_size: int, comp_size: int)`: Track network performance