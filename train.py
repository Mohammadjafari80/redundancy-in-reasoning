# train.py

import argparse
import logging
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import wandb
from utils import tokenize_function

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 model for code generation.")
    parser.add_argument('--task', type=str, default='gsm8k', help='Task/dataset to use for training.')
    parser.add_argument('--model_name_or_path', type=str, default='gpt2-medium', help='Pretrained model or path.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the fine-tuned model.')
    parser.add_argument('--include_reasoning',  type=str, default='False', help='Include reasoning in training data.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--injection_probability', type=float, default=0.0, help='Probability of Injecting a redundant sentence.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=4, help='Training batch size per device.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for training.')
    parser.add_argument('--use_wandb',  type=str, default='False', help='Log training metrics to Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='gpt2-finetuning', help='Weights & Biases project name.')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length for tokenization.')
    return parser.parse_args()

def main():
    args = parse_args()

    args.include_reasoning = args.include_reasoning.lower() == 'true'
    args.use_wandb = args.use_wandb.lower() == 'true'
    
    # Dynamically set output_dir based on parameters
    reasoning_suffix = "reasoning" if args.include_reasoning else "no_reasoning"
    inj_prob_suffix = f"inj_{args.injection_probability:.1f}".replace('.', '_')
    args.output_dir = f"./results/{reasoning_suffix}_{inj_prob_suffix}"
        
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Load and process the dataset based on the task
    if args.task == 'gsm8k':
        from gsm8k import GSM8K
        dataset_loader = GSM8K(
            split='train',
            include_answer=True,
            include_reasoning=args.include_reasoning,
            p=args.injection_probability,
            seed=42
        )
        dataset = dataset_loader.dataset
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.config.pad_token_id = tokenizer.eos_token_id

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Using device: {device}")

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=['text'],
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

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
        report_to=["wandb"] if args.use_wandb else [],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(args.output_dir)

    if args.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()