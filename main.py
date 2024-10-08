import argparse
import torch
from transformers import AutoModel, AutoTokenizer, OPTForCausalLM

from model import CustomLLM
from dataset import TextDataset, create_dataloaders
from training import train_llm


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
                        help="Path to the data file")
    parser.add_argument("--bottleneck-size", type=int, default=256,
                        help="Bottleneck size for the factorized embedding")
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
    parser.add_argument("--eval-split", type=float, default=0.1,
                        help="Proportion of data to use for evaluation")

    args = parser.parse_args()

    # Load models
    he_en_model = AutoModel.from_pretrained(args.he_en_model)
    en_he_model = AutoModel.from_pretrained(args.en_he_model)
    llm_model = OPTForCausalLM.from_pretrained(args.llm_model)

    # Use the tokenizer from the Hebrew-English model
    tokenizer = AutoTokenizer.from_pretrained(args.he_en_model)

    # Create dataset
    dataset = TextDataset(args.data_file, tokenizer, eval_split=args.eval_split)

    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(dataset, args.batch_size)

    # Create custom LLM
    custom_llm = CustomLLM(he_en_model, en_he_model, llm_model, len(tokenizer), args.bottleneck_size)

    # Train the model
    train_llm(custom_llm, train_dataloader, eval_dataloader, he_en_model, tokenizer,
              num_epochs=args.num_epochs,
              learning_rate=args.learning_rate,
              device=args.device,
              log_dir=args.log_dir,
              save_dir=args.save_dir)


if __name__ == "__main__":
    main()
