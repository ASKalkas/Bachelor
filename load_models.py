from transformers import AutoModelForCausalLM, AutoTokenizer

# Download Hermes model and tokenizer without loading into memory
print("Hermes")
AutoModelForCausalLM.from_pretrained(
    "NousResearch/Hermes-3-Llama-3.1-8B",
    device_map=None,  # Prevents loading onto any device
    torch_dtype=None,  # Prevents setting data type
    trust_remote_code=True,
    local_files_only=False,  # Ensures it downloads if not already present
)

AutoTokenizer.from_pretrained(
    "NousResearch/Hermes-3-Llama-3.1-8B",
    trust_remote_code=True,
    local_files_only=False
)

# Download Phi model and tokenizer without loading into memory
print("Phi")
AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct",
    device_map=None,
    torch_dtype=None,
    trust_remote_code=True,
    local_files_only=False,
)

AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    trust_remote_code=True,
    local_files_only=False
)

# Download Qwen model and tokenizer without loading into memory
print("Qwen")
AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    device_map=None,
    torch_dtype=None,
    trust_remote_code=True,
    local_files_only=False,
)

AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    local_files_only=False
)