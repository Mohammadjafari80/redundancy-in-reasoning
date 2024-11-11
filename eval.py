# eval.py

import argparse
import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import wandb
from utils import (
    extract_final_answer_from_code,
    generate_answer_single_shot,
    evaluate_single_shot,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned GPT-2 model.")
    parser.add_argument('--task', type=str, default='gsm8k', help='Task/dataset to use for evaluation.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model.')
    parser.add_argument('--include_reasoning', action='store_true', help='Include reasoning in the evaluation prompts.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation.')
    parser.add_argument('--num_beams', type=int, default=3, help='Number of beams for beam search.')
    parser.add_argument('--use_wandb', action='store_true', help='Log evaluation metrics to Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='gpt2-evaluation', help='Weights & Biases project name.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use for evaluation.')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Load and process the dataset based on the task
    if args.task == 'gsm8k':
        from gsm8k import GSM8K
        dataset_loader = GSM8K(
            split='test',
            include_answer=False,
            include_reasoning=args.include_reasoning,
        )
        dataset = dataset_loader.dataset
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.config.pad_token_id = tokenizer.eos_token_id

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Using device: {device}")

    # Generate answers and evaluate
    dataset = dataset.map(
        lambda x: generate_answer_single_shot(
            x, tokenizer, model, device, args.num_beams, args.temperature
        )
    )
    dataset = dataset.map(evaluate_single_shot)

    # Calculate accuracy
    correct_single_shot = 0
    total = 0
    for example in dataset:
        if example['predicted_answer_single_shot'] == example['final_answer']:
            correct_single_shot += 1
        total += 1

    accuracy_single_shot = correct_single_shot / total
    print(f"Single-Shot Accuracy: {accuracy_single_shot:.2%}")

    if args.use_wandb:
        wandb.log({'accuracy_single_shot': accuracy_single_shot})
        wandb.finish()

if __name__ == '__main__':
    main()