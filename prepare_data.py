import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_token = "[HF_TOKEN]"

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_representation_cache(data, output_path):
    torch.save(data, output_path)

@torch.no_grad()
def extract_encoder_representations(json_path, model_name, cache_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load data
    data = load_jsonl(json_path)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device)
    model.eval()

    all_data_with_repr = []

    for entry in tqdm(data):
        prompt = entry["prompt"]
        if prompt is None:
            # Skip or create a placeholder if desired
            continue

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        # Forward pass
        outputs = model.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # Take the last hidden state (encoder output equivalent)
        hidden_states = outputs.hidden_states[-1].squeeze(0).cpu()  # Shape: [seq_len, hidden_dim]

        # Add representation
        entry_with_repr = dict(entry)
        entry_with_repr["representation"] = hidden_states.numpy().tolist()  # or use torch.save later
        all_data_with_repr.append(entry_with_repr)

    # Save final result
    save_representation_cache(all_data_with_repr, cache_path)
    print(f"Saved cached representations to {cache_path}")



extract_encoder_representations(
    json_path="./data/populated_artifacts_sim/PAIR/llama-2-7b-chat-hf.jsonl",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    cache_path="./cache/llama2_actsim_representations.pt"
)

