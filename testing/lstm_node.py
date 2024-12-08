from w5xde.w5xde import TrainingNode
from custom_functions import Model

if __name__ == "__main__":
    model = Model(30522) # 30522 is vocab size
    node = TrainingNode(model, secure=False)
    node.train()