import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from transformers import AutoTokenizer
import numpy as np
import os

class SmartCalibrationDataReader(CalibrationDataReader):
    def __init__(self, tokenizer, model_path):
        self.tokenizer = tokenizer
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in session.get_inputs()]
        self.data = iter([
            "人工智能是计算机科学的一个分支。",
            "什么是深度学习？",
            "北京的天气预报是什么？",
            "Python 是一种广泛使用的编程语言。"
        ])

    def get_next(self):
        text = next(self.data, None)
        if text is None: return None
        
        # [关键修复] 强制对齐到导出时的 dummy_input 长度 32
        inputs = self.tokenizer(
            text, 
            return_tensors="np", 
            padding='max_length', 
            max_length=32, 
            truncation=True
        )
        
        ort_input = {}
        for name in self.input_names:
            if name in inputs:
                ort_input[name] = inputs[name].astype(np.int64)
        return ort_input

# 主程序
model_fp32 = "model_inf/qwen3_fp32.onnx"
model_int8 = "qwen3_int8.onnx"

tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B", trust_remote_code=True)
dr = SmartCalibrationDataReader(tokenizer, model_fp32)

print("--- Starting Quantization ---")

quantize_static(
    model_input=model_fp32,
    model_output=model_int8,
    calibration_data_reader=dr,
    quant_format=onnxruntime.quantization.QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    use_external_data_format=True, # 处理大模型权重限制
    per_channel=True,               # 提高量化精度，减少乱码
    reduce_range=True               # 防止某些 CPU 架构下的溢出
)
print(f"✅ Quantization Complete!")