import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, MarianMTModel, MarianTokenizer
)
from pathlib import Path
import json

from transformer1 import Transformer1, HebrewDataset, CustomTrainer, generate_example


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hebrew to English LLM adapter')
    parser.add_argument("--he-en-model", type=str, default="Helsinki-NLP/opus-mt-tc-big-he-en",
                        help="Name or path of the Hebrew-English model")
    parser.add_argument("--llm-model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Name or path of the LLM model")
    parser.add_argument('--train_data', type=str,
                        help='Path to training data TXT file with Hebrew sentences (one per line)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (0-1)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the model')
    parser.add_argument('--validate_every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')

    parser.add_argument('--generate', action='store_true',
                        help='Run in generation mode instead of training')
    parser.add_argument('--input_text', type=str,
                        help='Hebrew text to generate from (when using --generate)')
    parser.add_argument('--model_path', type=str,
                        help='Path to saved model checkpoint (when using --generate)')
    return parser.parse_args()


def load_data(file_path, val_split=0.1, seed=42):
    """Load sentences from a text file and optionally split into train/val"""
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Read and clean sentences
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(sentences)} sentences from {file_path}")

    # If no validation split requested, return all sentences for training
    if val_split <= 0:
        return sentences, None

    # Shuffle and split
    indices = torch.randperm(len(sentences)).tolist()
    split_idx = int(len(sentences) * (1 - val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_sentences = [sentences[i] for i in train_indices]
    val_sentences = [sentences[i] for i in val_indices]

    return train_sentences, val_sentences


def main():
    args = parse_args()

    if args.generate:
        if not args.input_text or not args.model_path:
            raise ValueError("Both --input_text and --model_path are required when using --generate")
        generate_example(args.model_path, args.input_text, args.device)
        return

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args for reproducibility
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Load models and tokenizers
    print("Loading models and tokenizers...")
    he_en_model = MarianMTModel.from_pretrained(args.he_en_model)
    he_en_tokenizer = MarianTokenizer.from_pretrained(args.he_en_model)

    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model)

    # Create config for the adapter model
    print("Creating adapter model...")

    adapter_model = Transformer1(
        he_en_model=he_en_model,
        llm_model=llm_model,
    )

    # Load and split data
    print("Loading and splitting data...")
    train_sentences, val_sentences = load_data(
        args.train_data,
        val_split=args.val_split,
        seed=args.seed
    )

    print(f"Training on {len(train_sentences)} sentences")
    if val_sentences:
        print(f"Validating on {len(val_sentences)} sentences")

    # Save some example sentences for reference
    examples = {
        'train': train_sentences[:5],
        'val': val_sentences[:5] if val_sentences else []
    }
    with open(output_dir / 'data_examples.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    # Create datasets
    train_dataset = HebrewDataset(
        hebrew_sentences=train_sentences,
        he_en_tokenizer=he_en_tokenizer,
        max_length=args.max_length
    )

    val_dataset = None
    if val_sentences:
        val_dataset = HebrewDataset(
            hebrew_sentences=val_sentences,
            he_en_tokenizer=he_en_tokenizer,
            max_length=args.max_length
        )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = CustomTrainer(
        model=adapter_model,
        he_en_model=he_en_model,
        llm_model=llm_model,
        he_en_tokenizer=he_en_tokenizer,
        llm_tokenizer=llm_tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=device
    )

    # Train the model
    print("Starting training...")
    try:
        train_losses, val_losses = trainer.train(
            num_epochs=args.epochs,
            validate_every=args.validate_every
        )

        # Save the model and training history
        print("Saving model and training history...")
        torch.save({
            'model_state_dict': adapter_model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'args': vars(args)
        }, output_dir / 'adapter_model.pt')

        training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses if val_losses else []
        }

        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2)

        print("Training complete!")
        print(f"Model and training history saved to {output_dir}")

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving checkpoint...")
        torch.save({
            'model_state_dict': adapter_model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'args': vars(args)
        }, output_dir / 'adapter_model_interrupted.pt')
        print("Checkpoint saved!")


if __name__ == '__main__':
    main()