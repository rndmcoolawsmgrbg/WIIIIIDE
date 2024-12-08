from transformers import AutoTokenizer
from w5xde import CentralServer
from custom_functions import Model, TextDataset

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') # too lazy rn
    model = Model(tokenizer.vocab_size)
    texts = open("dataset.txt").read().split("\n") # too lazy rn, again. replace w/ ur own stuff

    print(f"Dataset: {texts[:2]}...")

    dataset = TextDataset(texts, tokenizer)


    server = CentralServer(model=model,
        dataset=dataset,
        batch_size=128,
        checkpoint_dir="checkpoints",
        ip="0.0.0.0", # run on all interfaces
        port=5555,
        secure=False
    )
    server.start()