from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Check device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",             # Normalized Float 4 (optimal for LLMs)
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 for speed
    bnb_4bit_use_double_quant=True         # Double quantization for better accuracy
)

# Model name
model_name = "NousResearch/Llama-2-7b-chat-hf"

print("Loading model in 4-bit...")
# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",              # Automatically distribute across available devices
    trust_remote_code=True
)

print("Loading tokenizer...")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Model successfully loaded in 4-bit!")
print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")


# ============================================
# SAVE QUANTIZED MODEL
# ============================================
save_directory = "./llama2-7b-4bit-quantized"

print(f"\nSaving quantized model to {save_directory}...")

# Save the model
model.save_pretrained(
    save_directory,
    safe_serialization=True  # Use safetensors format (recommended)
)

# Save the tokenizer
tokenizer.save_pretrained(save_directory)

# Save the quantization config separately (optional but useful)
import json
quant_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True
}
with open(f"{save_directory}/quantization_config.json", "w") as f:
    json.dump(quant_config, f, indent=2)

print(f"✅ Model saved successfully!")
print(f"   - Model weights: {save_directory}")
print(f"   - Tokenizer: {save_directory}")
print(f"   - Quantization config: {save_directory}/quantization_config.json")
