import torch
import torch.nn as nn
from transformers import AutoTokenizer
from custom_functions import Model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextGenerator:
    def __init__(self, model_path, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = tokenizer
        self.device = device

        self.model = Model(tokenizer.vocab_size)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()

    def generate(self, prompt, max_length=32, temperature=1, top_k=10):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature

                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)

                probs = torch.softmax(top_k_logits, dim=-1)

                next_token_idx = torch.multinomial(probs[0], 1)
                next_token = top_k_indices[0][next_token_idx]

                next_token = next_token.unsqueeze(0) 
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text

def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    generator = TextGenerator('trained_model.pth', tokenizer)
    
    prompt = "Come, Montague, for thou art"
    generated_text = generator.generate(prompt)
    logger.info(f"Prompt: {prompt}")

if __name__ == "__main__":
    main()