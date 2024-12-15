import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

dataset = load_dataset("gsm8k", "main")

test_data = dataset["test"]

def solve_problem(question, max_new_tokens=256):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Please act as a professional math teacher.\n"
        "Your goal is to accurately solve a math word problem.\n"
        "To achieve the goal, you have two jobs.\n"
        "# Write detailed solution to a Given Question.\n"
        "# Write the final answer to this question.\n"
        "You have two principles to do this.\n"
        "# Ensure the solution is step-by-step.\n"
        "# Ensure the final answer is just a number (float or integer).\n"
        "Your output should be in the following format:\n"
        "SOLUTION: <your detailed solution to the given question>\n"
        "FINAL ANSWER: <your final answer to the question with only an integer or float number>\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "FINAL ANSWER:" in generated:
        answer_text = generated.split("FINAL ANSWER:")[-1].strip()
    else:
        answer_text = generated.strip()

    numbers_full = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_text)
    if numbers_full:
        return numbers_full[-1]
    return answer_text

correct = 0
total = 0
num_eval = 100  # 일부 문제만 평가

for i, example in enumerate(tqdm(test_data, total=len(test_data))):
    # 주석 처리시 전체에 대해 평가
    # if i >= num_eval:
    #     break
    question = example["question"]
    gold_answer = example["answer"]

    import re
    gold_numbers = re.findall(r"####\s*([0-9]+)", gold_answer)
    if not gold_numbers:
        continue
    gold = gold_numbers[-1]

    pred = solve_problem(question)
    total += 1
    if pred == gold:
        correct += 1
        print("Correct!")
    else: print("Wrong..")

accuracy = correct / total if total > 0 else 0
print(f"Evaluated on {total} samples.")
print(f"Accuracy: {accuracy * 100:.2f}%")