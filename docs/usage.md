# Using the W5XDE Distributed Training System

## Basic Setup

### Server Setup
```python
from w5xde import CentralServer

server = CentralServer(
    model=your_model,
    dataset=your_dataset,
    batch_size=32,
    ip="localhost",
    port=5555
)
server.start()
```

### Training Node Setup
```python
from w5xde import TrainingNode

node = TrainingNode(
    model=your_model,
    server_address=('localhost', 5555),
    collect_metrics=True,  # Optional: enable performance metrics
    compress_gradients=False  # Recommended: keep compression disabled
)
node.train()
```

## Advanced Features

### Performance Metrics
Enable detailed performance tracking:
```python
def track_loss(loss, batch_id):
    print(f"Batch {batch_id}: loss = {loss:.4f}")

def track_network(sent, received, comp_time, net_time, orig_size, comp_size):
    print(f"Network metrics - Sent: {sent}, Received: {received}")
    print(f"Compression ratio: {orig_size/comp_size:.2f}x")

node = TrainingNode(
    model=your_model,
    collect_metrics=True
)
node.train(loss_callback=track_loss, network_callback=track_network)
```

### Gradient Compression (Optional)
Enable gradient compression (note: not recommended for most cases):
```python
node = TrainingNode(
    model=your_model,
    compress_gradients=True,
    batch_gradients=True
)
```

## Recent Changes and Optimizations

1. **Socket Configuration**
   - Added TCP_NODELAY for faster small messages
   - Increased buffer sizes to 1MB
   - Enabled TCP keepalive

2. **Memory Management**
   - Implemented pre-allocated buffers
   - Added zero-copy operations where possible
   - Optimized gradient processing

3. **Compression System**
   - Added optional gradient compression
   - Implemented error feedback
   - Added adaptive compression levels

4. **Network Protocol**
   - Optimized message sending/receiving
   - Added metrics collection
   - Improved error handling

## Best Practices

1. **General Usage**
   - Start with compression disabled
   - Enable metrics collection during testing
   - Use appropriate batch sizes for your network

2. **Network Configuration**
   - Use local testing first
   - Ensure proper port cleanup between runs
   - Monitor network metrics during training

3. **Resource Management**
   - Monitor memory usage with metrics
   - Clean up resources properly after training
   - Use appropriate timeouts for your network 