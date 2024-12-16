import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# dataset = load_dataset("competition_math")
# print(len(dataset["test"]))

class EvaluateSLM:
    def __init__(self, model_name, prompt_bank, dataset_configs):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.prompt_template = prompt_bank.get(model_name)
        self.dataset_configs = dataset_configs
    
    def solve_problem(self, question, max_new_tokens = 1024):
        prompt = self.prompt_template.format(question = question)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False
            )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # cat generated response
        print("generated: \n", generated)

        if "FINAL ANSWER:" in generated:
            answer_text = generated.split("FINAL ANSWER:")[-1].strip()
        else:
            answer_text = generated.strip()

        numbers_full = re.findall(r"[-+]?\d+(?:\.\d+)?", answer_text)
        if numbers_full:
            return numbers_full[-1]
        
        return answer_text
    
    def evaluate_dataset(self, dataset_name, split="test"):
        # 데이터셋 구성 가져오기
        dataset_config = self.dataset_configs.get(dataset_name)
        question_key = dataset_config.get("question_key")
        answer_key = dataset_config.get("answer_key")
        answer_pattern = dataset_config.get("answer_pattern")

        dataset = load_dataset(dataset_name, 'main')
        test_data = dataset[split]
        
        print("Dataset load success, length is:", len(test_data))

        correct = 0
        total = 0

        for example in tqdm(test_data, total=len(test_data)):
            question = example[question_key]
            gold_answer = example[answer_key]

            gold_numbers = re.findall(answer_pattern, gold_answer)
            if not gold_numbers:
                continue
            gold = gold_numbers[-1].strip()

            pred = self.solve_problem(question)
            total += 1
            if pred == gold:
                correct += 1
                print(f"Correct! Q: {question} | Pred: {pred} | Gold: {gold}")
            else:
                print(f"Wrong.. Q: {question} | Pred: {pred} | Gold: {gold}")

        accuracy = correct / total if total > 0 else 0
        print(f"Evaluated on {total} samples.")
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy    
    
    # def evaluate_gsm8k(self, question, max_new_tokens = 256):
    #     prompt = (
    #     "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    #     "Please act as a professional math teacher.\n"
    #     "Your goal is to accurately solve a math word problem.\n"
    #     "To achieve the goal, you have two jobs.\n"
    #     "# Write detailed solution to a Given Question.\n"
    #     "# Write the final answer to this question.\n"
    #     "You have two principles to do this.\n"
    #     "# Ensure the solution is step-by-step.\n"
    #     "# Ensure the final answer is just a number (float or integer).\n"
    #     "Your output should be in the following format:\n"
    #     "SOLUTION: <your detailed solution to the given question>\n"
    #     "FINAL ANSWER: <your final answer to the question with only an integer or float number>\n"
    #     "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    #     f"{question}\n"
    #     "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # )
    
def evaluate_models_on_datasets(models, datasets, prompt_templates, dataset_configs, split="test"):
    results = {}
    for model_name in models:
        print(f"\nLoading model: {model_name}")
        evaluator = EvaluateSLM(model_name, prompt_templates, dataset_configs)
        for dataset_name in datasets:
            print(f"\nEvaluating on dataset: {dataset_name}")
            accuracy = evaluator.evaluate_dataset(dataset_name, split=split)
            results[(model_name, dataset_name)] = accuracy
    return results
        
if __name__ == "__main__":
    models = [
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    
    datasets = [
        # "hendrycks/competition_math",
        "openai/gsm8k",
    ]
    
    prompt_templates = {
        "meta-llama/Llama-3.2-3B-Instruct" : (
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
            "Make sure not to include any character other than integer or float number in FINAL ANSWER."
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            "{question}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    }
    
    dataset_configs = {
        "openai/gsm8k": {
            "question_key": "question",
            "answer_key": "answer",
            "answer_pattern": r"####\s*([0-9]+)"
        },
        "hendrycks/competition_math": {
            "question_key": "problem",
            "answer_key": "solution",
            "answer_pattern": r"\\boxed{([^}]*)}"
        }
    }
    
    results = evaluate_models_on_datasets(models, datasets, prompt_templates, dataset_configs)
    print("\nFinal Results:")
    for (model, dataset), accuracy in results.items():
        print(f"Model: {model} | Dataset: {dataset} | Accuracy: {accuracy * 100:.2f}%")