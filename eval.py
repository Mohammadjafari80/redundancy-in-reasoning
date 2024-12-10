import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
from utils import (
    extract_final_answer_from_code,
    generate_answer,
    evaluate,
)
import json
import re
import os
from peft import PeftModel, LoraConfig, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned GPT-2 model.")
    parser.add_argument('--task', type=str, default='gsm8k', help='Task/dataset to use for evaluation.', choices=['gsm8k_qa', 'gsm8k_code'])
    parser.add_argument('--model_name', type=str, required=True, help='Name of the fine-tuned model.', default='gpt2-medium')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the fine-tuned model.')
    parser.add_argument('--injection_probability', type=float, default=0.0, help='Probability of Injecting a redundant sentence.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation.')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search.')
    parser.add_argument('--use_wandb', type=str, default='False', help='Log evaluation metrics to Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='gsm8k-evaluation', help='Weights & Biases project name.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use for evaluation.')
    parser.add_argument('--few_shot', type=str, default='False', help='Enable few-shot evaluation.')
    parser.add_argument('--num_shots', type=int, default=8, help='Number of shots for fewshot evaluation.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--use_lora', type=str, default='False', help='Use LoRA for evaluation.')
    return parser.parse_args()

def main():
    args = parse_args()

    args.use_wandb = args.use_wandb.lower() == 'true'
    args.few_shot = args.few_shot.lower() == 'true'
    args.use_lora = args.use_lora.lower() == 'true'

    print(args)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    filename = f"{args.task}_no-cot_{args.model_name}_inj{args.injection_probability}_temp{args.temperature}_beams{args.num_beams}_split{args.split}"
    if args.few_shot:
        filename += f"_fewshot{args.num_shots}"

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args), name=filename, tags=[args.model_name, args.task, 'cot'])

    # Load and process the dataset based on the task
    if args.task.startswith('gsm8k'):
        from gsm8k import GSM8K
        template = args.task.split('_')[-1]
        dataset_loader = GSM8K(
            split=args.split,
            include_answer=False,
            include_reasoning=False,
            p=args.injection_probability,
            few_shot=args.few_shot,
            num_shots=args.num_shots,
            seed=42,
            template=template,
            cot=True
        )
        dataset_loader.view_example(0)
        dataset = dataset_loader.dataset
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Initialize tokenizer and model
    print(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')

    if args.model_path is not None:
        lora_suffix = "lora" if args.use_lora else ""
        # model_path = f"results/{args.model_name}/lora/{args.model_path}"
        model_path = args.model_path
        if args.use_lora:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype='auto')
            model = PeftModel.from_pretrained(model, model_path)
            logger.info(f"LoRA model loaded from {model_path}")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
            logger.info(f"Full model loaded from {model_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype='auto')
        logger.info(f"Model loaded from {args.model_name}")

    # Set padding token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Update model configuration for padding
    model.config.pad_token_id = tokenizer.pad_token_id

    max_length = 512

    if args.few_shot:
        max_length = max_length = min(model.config.max_position_embeddings, 2048)

    print(f"Max length: {max_length}")

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model = torch.compile(model)
    logger.info(f"Using device: {device}")

    # Generate answers and evaluate
    dataset = dataset.map(
        lambda x: generate_answer(
            x, tokenizer, model, device, args.num_beams, args.temperature, max_length=max_length, model_name=args.model_name
        ),
        batched=True,
        batch_size=args.batch_size,
    )

    dataset = dataset.map(lambda x: evaluate(x, args.task))

    def evaluate_accuracy(example):
        predicted_answer = example['predicted_answer']
        final_answer = example['final_answer']
        example['correct'] = False
        try:
            pred_value = float(str(predicted_answer).replace(',', ''))
            final_value = float(str(final_answer).replace(',', ''))
            if abs(pred_value - final_value) < 1e-3:
                example['correct'] = True
        except (ValueError, TypeError):
            if predicted_answer is not None and predicted_answer.strip() == final_answer.strip():
                example['correct'] = True
        return example

    dataset = dataset.map(evaluate_accuracy)

    # Calculate accuracy
    correct = sum(1 for example in dataset if example['correct'])
    total = len(dataset)
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.2%}")

    if args.use_wandb:
        wandb.log({'accuracy': accuracy})
        wandb.finish()

    # Create a meaningful filename based on args
    def sanitize_filename(filename):
        # Remove illegal characters for filenames
        return re.sub(r'[<>:"/\\|?*]', '', filename)

    filename = sanitize_filename(filename)
    filename += ".jsonl"

    # Remove unwanted columns (e.g., tensors)
    columns_to_remove = [col for col in dataset.column_names if isinstance(dataset[col][0], (torch.Tensor, list))]
    dataset = dataset.remove_columns(columns_to_remove)

    # Save the dataset to a JSONL file
    output_dir = "json-results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    dataset.to_json(output_path)
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
