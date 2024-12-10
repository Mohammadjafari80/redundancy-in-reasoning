import argparse
import logging
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from contrastive_trainer import ContrastiveTrainer
import wandb
from utils import tokenize_function, ContrastivePreprocess
from peft import get_peft_model, LoraConfig, TaskType
import os
import json

# Function to print the number of trainable parameters
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"All parameters: {all_params}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / all_params:.2f}%")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for Reasoning with LoRA support.")
    parser.add_argument('--task', type=str, default='gsm8k', help='Task/dataset to use for training.')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2-medium', help='Pretrained model or path.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the fine-tuned model.')
    parser.add_argument('--include_reasoning', type=str, default='False', help='Include reasoning in training data.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--injection_probability', type=float, default=0.0, help='Probability of Injecting a redundant sentence.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='Training batch size per device.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for training.')
    parser.add_argument('--use_wandb', type=str, default='False', help='Log training metrics to Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='llm-finetuning', help='Weights & Biases project name.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization.')
    parser.add_argument('--use_lora', type=str, default='False', help='Use LoRA for fine-tuning.')
    parser.add_argument('--lora_rank', type=int, default=32, help='Rank of the low-rank matrices for LoRA.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Scaling factor for the low-rank matrices for LoRA.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='Dropout rate for LoRA layers.')
    parser.add_argument('--lora_target_modules', type=str, help='Modules to apply LoRA.')
    
    parser.add_argument('--contrastive_weight', type=float, default=0.0, help='Contrastive Loss weight')
    parser.add_argument('--lm_weight', type=float, default=1.0, help='Language Modeling Loss weight')
    parser.add_argument('--completion_only', type=str, default='False', help='Whether to finetune on whole input or just the completion.')

    # contrastive_weight
    return parser.parse_args()

def main():
    args = parse_args()

    args.include_reasoning = args.include_reasoning.lower() == 'true'
    args.use_wandb = args.use_wandb.lower() == 'true'
    args.use_lora = args.use_lora.lower() == 'true'
    args.completion_only = args.completion_only.lower() == 'true'
    args.lora_target_modules = None if args.lora_target_modules in ['None', None] else args.lora_target_modules
    
    # Dynamically set output_dir based on parameters
    reasoning_suffix = "reasoning" if args.include_reasoning else "no_reasoning"
    inj_prob_suffix = f"inj_{args.injection_probability:.1f}".replace('.', '_')
    lora_suffix = "lora" if args.use_lora else "full-finetune"
    completion_only_suffix = "Completion" if args.completion_only else "All"
    contrastive_suffix = 'ContLoss' + str(args.contrastive_weight).replace('.', '-')
    lm_suffix = 'LMLoss' + str(args.lm_weight).replace('.', '-')
    args.output_dir = f"./results/{args.model_name_or_path}/{lora_suffix}/{completion_only_suffix}/{reasoning_suffix}_{inj_prob_suffix}_{contrastive_suffix}_{lm_suffix}_cot"

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args), tags=[args.model_name_or_path, args.task, 'cot', contrastive_suffix, completion_only_suffix])

    # Load and process the dataset based on the task
    if args.task.startswith('gsm8k'):
        from gsm8k import GSM8K
        template = args.task.split('_')[-1]
        dataset_loader = GSM8K(
            split='train',
            include_answer=True,
            include_reasoning=args.include_reasoning,
            p=args.injection_probability,
            seed=42,
            few_shot=False,
            template=template,
            cot=True
        )
        dataset_loader.view_example(0)
        dataset = dataset_loader.dataset
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype='auto')
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA if enabled
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        print("LoRA configuration applied for training.")
        print_trainable_parameters(model)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        logging_dir='./logs',
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        logging_first_step=True,
        report_to=["wandb"] if args.use_wandb else [],
    )

    print(dataset[0])
    dataset = ContrastivePreprocess(dataset, tokenizer).ds
    
    # Initialize Trainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        contrastive_weight=args.contrastive_weight,
        lm_weight=args.lm_weight,
        completion_only=args.completion_only,
        tkn=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    if args.use_lora:
        model.save_pretrained(args.output_dir)
        print(f"LoRA model and configuration saved to {args.output_dir}")
    else:
        trainer.save_model(args.output_dir)
        print(f"Full model saved to {args.output_dir}")

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
