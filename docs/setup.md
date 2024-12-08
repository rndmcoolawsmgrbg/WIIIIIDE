# Usage

## Installation

All you really need is Python 3.8+ and PyTorch. That's seriously it, so it's really minimal.

```bash
git clone https://github.com/rndmcoolawsmgrbg/wiiiiide.git
cd wiiiiide
pip install -r requirements.txt
```

## Getting Started

The concept is to throw the w5xde.py file into your project and use it as a module. 

So, as a rule of thumb, you should have a `server.py` and a `node.py` in your project. These can serve as the central server and training nodes, respectively.

Your server should contain imports for your dataset class, and model. The node, only requires the model (Make sure it's the same model as the server, or you'll have a bad time).

Setting up a server is as simple as 3 extra lines of code.

```python
from w5xde import CentralServer

server = CentralServer(model, dataset)
server.start()
```

Keep in mind this is the minimal setup. You can specify a lot more parameters, such as batch size, IP, port, and whether or not to use encryption.

Setting up a node is just as simple.

```python
from w5xde import TrainingNode

# Simple usage without metrics
node = TrainingNode(MODEL_CLASS)
node.train()

# Or with metrics collection for debugging/optimization
def track_loss(loss, batch_id):
    print(f"Batch {batch_id}: loss = {loss:.4f}")

def track_network(sent, received, comp_time, net_time, orig_size, comp_size):
    print(f"Network metrics - Sent: {sent}, Received: {received}, Compression ratio: {orig_size/comp_size:.2f}x")

node = TrainingNode(MODEL_CLASS, collect_metrics=True)
node.train(loss_callback=track_loss, network_callback=track_network)
```

And that's it! WIIIIIDE will take care of the rest. (For more info, check out the [arguments and classes](classes.md))