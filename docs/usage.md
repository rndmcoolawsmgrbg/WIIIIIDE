# Using the W5XDE Distributed Training System

## Basic Setup

### Server Setup
```python
import torch
from w5xde import CentralServer

# Your PyTorch model
class YourModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ... your model definition ...

# Your PyTorch dataset
class YourDataset(torch.utils.data.Dataset):
    def __init__(self):
        # ... your dataset definition ...

# Initialize server
server = CentralServer(
    model=YourModel(),
    dataset=YourDataset(),
    batch_size=32,
    ip="0.0.0.0",  # Allow external connections
    port=5555
)
server.start()
```

### Training Node Setup
```python
import torch
from w5xde import TrainingNode

# Must match server's model architecture
class YourModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # ... same model definition as server ...

# Basic setup
node = TrainingNode(
    model=YourModel(),
    server_address=('server_ip', 5555),
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
    print(f"Network stats:")
    print(f"  Throughput: {(sent + received) / (1024 * net_time):.2f} KB/s")
    print(f"  Compression time: {comp_time:.3f}s")
    print(f"  Network time: {net_time:.3f}s")
    print(f"  Compression ratio: {orig_size/comp_size:.2f}x")

node = TrainingNode(
    model=YourModel(),
    server_address=('server_ip', 5555),
    collect_metrics=True
)
node.train(loss_callback=track_loss, network_callback=track_network)
```

### Gradient Compression (Optional)
Enable gradient compression (note: not recommended for most cases):
```python
node = TrainingNode(
    model=YourModel(),
    compress_gradients=True,
    batch_gradients=True
)
```

## Recent Changes and Optimizations

1. **Socket Configuration**
   - Added TCP_NODELAY for faster small messages
   - Increased buffer sizes to 16MB
   - Enabled TCP keepalive
   - Platform-specific optimizations

2. **Memory Management**
   - Implemented pre-allocated buffers
   - Added zero-copy operations where possible
   - Optimized gradient processing
   - Direct tensor serialization

3. **Compression System**
   - Added optional gradient compression
   - Implemented error feedback
   - Added adaptive compression levels
   - Efficient tensor handling

4. **Network Protocol**
   - Optimized message sending/receiving
   - Added metrics collection
   - Improved error handling
   - Cross-platform compatibility

## Platform-Specific Considerations

### Windows
- TCP optimizations are automatically configured
- Buffer sizes are set appropriately
- TCP FastOpen is enabled on Windows 10 1607+

### Linux
- Additional TCP optimizations available
- TCP QuickACK and thin-stream optimizations enabled
- Larger buffer sizes supported

### MacOS
- Platform-specific TCP optimizations
- Adjusted buffer management
- Compatible network settings

## Best Practices

1. **General Usage**
   - Start with compression disabled
   - Enable metrics collection during testing
   - Use appropriate batch sizes for your network
   - Ensure model definitions match between server and nodes

2. **Network Configuration**
   - Use local testing first
   - Ensure proper port cleanup between runs
   - Monitor network metrics during training
   - Check firewall settings for port 5555 (or chosen port)

3. **Resource Management**
   - Monitor memory usage with metrics
   - Clean up resources properly after training
   - Use appropriate timeouts for your network
   - Adjust batch size based on available memory

## Common Issues and Solutions

1. **Connection Problems**
   - Verify server IP is accessible
   - Check firewall settings
   - Ensure model definitions match
   - Verify network connectivity

2. **Performance Issues**
   - Enable metrics collection for debugging
   - Verify network capacity
   - Check system resources
   - Monitor compression ratios

3. **Memory Warnings**
   - Adjust batch size
   - Monitor system memory
   - Check for memory leaks in custom models
   - Verify tensor cleanup