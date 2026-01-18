import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

model_path = "qwen3_int8.onnx"
tokenizer_path = "./Qwen3-1.7B"

sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
input_nodes = [inp.name for inp in sess.get_inputs()]

def generate(prompt, max_tokens=30):
    # 1. 预处理并 Padding 到 32
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="np", padding='max_length', max_length=32, truncation=True)
    input_ids = inputs["input_ids"].astype(np.int64)
    
    # 记录当前真实文本的结尾索引
    current_pos = np.sum(inputs["attention_mask"]) - 1
    
    print(f"Qwen: ", end="", flush=True)
    
    for _ in range(max_tokens):
        # 2. 构造输入
        ort_inputs = {"input_ids": input_ids}
        if "attention_mask" in input_nodes:
            # 这里的 mask 也要保持长度 32
            ort_inputs["attention_mask"] = inputs["attention_mask"].astype(np.int64)

        # 3. 推理
        outputs = sess.run(None, ort_inputs)
        logits = outputs[0]
        
        # 4. 获取当前位置的预测值
        next_token = np.argmax(logits[0, current_pos, :])
        
        if next_token == tokenizer.eos_token_id or current_pos >= 31:
            break
            
        word = tokenizer.decode([next_token], skip_special_tokens=True)
        print(word, end="", flush=True)
        
        # 5. 更新序列：将新 token 填入下一个位置
        current_pos += 1
        input_ids[0, current_pos] = next_token
        if "attention_mask" in input_nodes:
            inputs["attention_mask"][0, current_pos] = 1
        
    print("\n")

while True:
    q = input("\nUser: ")
    if q.lower() == "exit": break
    generate(q)