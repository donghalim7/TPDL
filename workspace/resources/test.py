import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# dataset = load_dataset("gsm8k", "main")

# test_data = dataset["test"]

prompt = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "Answer the given questions.""<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "What is the capital of France?"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "ANSWER:"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 모델 출력 생성
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.0,
        do_sample=False
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Model output:\n", response)