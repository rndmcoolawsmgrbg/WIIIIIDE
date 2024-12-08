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

node = TrainingNode(MODEL_CLASS)
node.train()
```

You can also track training progress by providing a callback function:

```python
def track_progress(loss, batch_id):
    print(f"Batch {batch_id}: loss = {loss:.4f}")

node = TrainingNode(MODEL_CLASS)
node.train(loss_callback=track_progress)
```

And that's it! WIIIIIDE will take care of the rest. (For more info, check out the [arguments and classes](classes.md))