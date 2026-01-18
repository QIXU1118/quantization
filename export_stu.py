import os
# [CRITICAL] Set this BEFORE importing torch to ensure it takes effect
os.environ["TORCH_ONNX_FALLBACK_TO_LEGACY"] = "1" 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# ================= TODO 1: Implement Mask Patch =================
def mask_patch(*args, **kwargs):
    # 1. Parse Input Shape dynamically
    # Try to get input_shape from kwargs (transformers standard) or args
    input_shape = kwargs.get("input_shape", None)
    if input_shape is None and len(args) > 0:
        input_shape = args[0]
    
    # [Fix] Do not use hardcoded 32. Get dimension from tensor if possible
    if isinstance(input_shape, torch.Tensor):
        bsz, seq_len = input_shape.shape[0], input_shape.shape[1]
    elif input_shape is not None:
        bsz, seq_len = input_shape[0], input_shape[1]
    else:
        # Fallback only if absolutely necessary, but this path shouldn't be hit in tracing
        bsz, seq_len = 1, 32 

    dtype = kwargs.get("dtype", torch.float32)
    device = kwargs.get("device", torch.device("cpu"))

    # 2. Generate Mask (ONNX-friendly triu)
    # Create a full matrix of min values
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    # Keep upper triangle (causal mask logic)
    mask = torch.triu(mask, diagonal=1)
    
    # 3. Return 4D Tensor
    return mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)

# Apply Patch aggressively to all possible entry points
try:
    import transformers.modeling_attn_mask_utils as amu
    amu._prepare_4d_causal_attention_mask = mask_patch
    # Hook the general utility as well
    transformers.masking_utils.create_causal_mask = mask_patch
except ImportError:
    pass
print(">>> [Patch Applied] Mask logic forced to legacy compatible mode")


# ================= TODO 2: Model Wrapper =================
class Qwen3ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # 1. Call model with use_cache=False
        # 2. Return dictionary to False (optional, but cleaner for tracing)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True
        )
        return outputs.logits


# ================= Main Execution =================
# Use absolute path
model_path = os.path.abspath("/data2/xjp/Class_Project/Quantization/Qwen3-1.7B")
output_file = "qwen3_fp32.onnx"

print(f"--- Loading Model from {model_path} ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32, 
        device_map="cpu", 
        trust_remote_code=True,
        attn_implementation="eager", # Eager is strictly required for export
        local_files_only=True
    )
    base_model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit(1)

model_wrapper = Qwen3ONNXWrapper(base_model)

# Dummy inputs
dummy_input_ids = torch.ones((1, 32), dtype=torch.long)
dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

print(f"--- Exporting to {output_file} ---")

# ================= TODO 3: Export Configuration =================
with torch.no_grad():
    # [CRITICAL FIX]
    # We use a try-except block to handle different torch versions specific flags
    try:
        torch.onnx.export(
            model_wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_file,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=14,
            do_constant_folding=True,
            
            # This is the "Magic Flag" for PyTorch 2.0+ to prevent Dynamo errors
            # If this argument doesn't exist in your version, it will trigger the except block
            dynamo=False 
        )
    except TypeError:
        # Fallback for older/newer versions that don't support 'dynamo=False' kwarg
        # but respect the env var set at top
        print(">>> 'dynamo' arg not found, relying on env var fallback...")
        torch.onnx.export(
            model_wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_file,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size", 1: "seq_len"},
            },
            opset_version=14,
            do_constant_folding=True
        )

print(f"✅ Export Success!")