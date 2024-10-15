import argparse
import os
import traceback

import torch
from transformers import AutoTokenizer, OPTForCausalLM, MarianMTModel, MarianTokenizer

from model import CustomLLM
from dataset import create_dataloaders
from training import train_llm
from lr_finder import find_best_lr


def main():
    parser = argparse.ArgumentParser(description="Train a custom LLM model")
    parser.add_argument("--he-en-model", type=str, default="Helsinki-NLP/opus-mt-tc-big-he-en",
                        help="Name or path of the Hebrew-English model")
    parser.add_argument("--en-he-model", type=str, default="Helsinki-NLP/opus-mt-en-he",
                        help="Name or path of the English-Hebrew model")
    parser.add_argument("--llm-model", type=str, default="facebook/opt-350m",
                        help="Name or path of the LLM model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to the data file containing Hebrew sentences")
    parser.add_argument("--num-epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory to save log files")
    parser.add_argument("--save-dir", type=str, default="model_checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Proportion of data to use for training")
    parser.add_argument("--display-interval", type=int, default=100,
                        help="Display interval to log the last sentence in the batch")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint file to resume training from")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Path to a pretrained CustomLLM model to load and fine-tune")
    parser.add_argument("--find-lr", action="store_true", help="Run learning rate finder before training")
    parser.add_argument("--lr-plot-path", type=str, default="lr_finder_plot.png",
                        help="Path to save the learning rate finder plot")
    parser.add_argument("--generate", action="store_true", help="Run in generation mode")

    args = parser.parse_args()

    # Load models
    he_en_model = MarianMTModel.from_pretrained(args.he_en_model).to(args.device)
    en_he_model = MarianMTModel.from_pretrained(args.en_he_model).to(args.device)
    llm_model = OPTForCausalLM.from_pretrained(args.llm_model).to(args.device)

    # Load tokenizers
    tokenizer1 = MarianTokenizer.from_pretrained(args.he_en_model)
    tokenizer2 = AutoTokenizer.from_pretrained(args.llm_model)
    tokenizer3 = MarianTokenizer.from_pretrained(args.en_he_model)

    if args.pretrained_model:
        # Load the pretrained CustomLLM
        print(f"Loading pretrained model from {args.pretrained_model}")
        custom_llm = CustomLLM.load_pretrained(args.pretrained_model, he_en_model, en_he_model, llm_model, args.device)
    else:
        # Create a new CustomLLM
        custom_llm = CustomLLM(he_en_model, en_he_model, llm_model)

    # Move the model to the specified device
    custom_llm = custom_llm.to(args.device)
    if args.generate:
        print("Entering generation mode. Type 'quit' to exit.")
        while True:
            sentence = input("Enter a Hebrew sentence: ")
            if sentence.lower() == 'quit':
                break
            try:
                generated_ids = custom_llm.generate(sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, args.device, llm=llm_model)
                generated_text = tokenizer3.decode(generated_ids[0], skip_special_tokens=True)
                print(f"Generated text: {generated_text}")
            except Exception as e:
                print(f"An error occurred during generation: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
    else:
        # Load sentences from the data file
        with open(args.data_file, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        sentences = [line.strip() for line in sentences if line.strip()]

        # Create dataloaders
        train_dataloader, eval_dataloader = create_dataloaders(sentences, he_en_model, tokenizer1, tokenizer2,
                                                               tokenizer3,
                                                               batch_size=args.batch_size, train_split=args.train_split,
                                                               device=args.device)

        if args.find_lr:
            best_lr = find_best_lr(custom_llm, train_dataloader, args.device, args.lr_plot_path)
            print(f"Best learning rate found: {best_lr}")
            args.learning_rate = best_lr / 10  # Use a slightly lower learning rate for training

        # Set CUDA_LAUNCH_BLOCKING for better error reporting
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        try:
            # Train the model
            train_llm(custom_llm, train_dataloader, eval_dataloader, tokenizer1, tokenizer3,
                      num_epochs=args.num_epochs,
                      learning_rate=args.learning_rate,
                      device=args.device,
                      log_dir=args.log_dir,
                      save_dir=args.save_dir,
                      checkpoint=args.checkpoint,
                      display_interval=args.display_interval)
        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
