# Classes, Arguments, and Methods

## Classes

W3XDE only contains two classes. (well, for actual use!) The `CentralServer` and the `TrainingNode`.

### CentralServer

The `CentralServer` class is the central hub for all training data. It's responsible for distributing batches to all connected `TrainingNode` instances, and aggregating gradients from them.

#### Arguments

- `model` The model to train. Must be a PyTorch model class.
- `dataset` The dataset to train on. Must be a PyTorch dataset class.
- `batch_size` The size of the batches to distribute. Default is 16.
- `ip` The IP address to bind the server to. Default is "localhost". ("0.0.0.0" for all interfaces)
- `port` The port to bind the server to. Default is 5555.
- `secure` Whether or not to use encryption. Default is False.
- `checkpoint_interval` How often to save checkpoints. Default is 5.
  
#### Methods
- `start()` Starts the server. This is a blocking call, so it will not return until the server is stopped. 
  
### TrainingNode

The `TrainingNode` class is the client that connects to the `CentralServer` to receive batches and send gradients.

#### Arguments

- `model` The model to train. Must be a PyTorch model class.
- `server_address` A tuple of (ip, port) for the server connection. Default is ('localhost', 5555).
- `secure` Whether or not to use encryption. Default is False.
- `collect_metrics` Whether to collect network and compression metrics. Default is False.

#### Methods
- `train(loss_callback=None, network_callback=None)` Starts training the model.
  - `loss_callback`: Optional callback function that receives (loss_value, batch_id)
  - `network_callback`: Optional callback function that receives network metrics (sent_bytes, received_bytes, comp_time, net_time, orig_size, comp_size)