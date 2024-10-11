import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer, OPTForCausalLM, AutoTokenizer
from torch.nn import TransformerEncoderLayer
import argparse

from utils import print_progress_bar, setup_logger


class ModifiedLLM(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, input_ids, attention_mask=None):
        # Get the embeddings
        inputs_embeds = self.model.model.decoder.embed_tokens(input_ids)
        inputs_embeds = self.model.model.decoder.project_in(inputs_embeds)

        # Add positional embeddings
        pos_embeds = self.model.model.decoder.embed_positions(attention_mask, 0)
        hidden_states = inputs_embeds + pos_embeds

        # Return the hidden states after embeddings
        return hidden_states


class CustomLayer1(nn.Module):
    def __init__(self, he_en_hidden_size, llm_hidden_size, num_attention_heads, ffn_dim, dropout, custom_layers):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(he_en_hidden_size, llm_hidden_size),
            *[TransformerEncoderLayer(
                d_model=llm_hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="relu",
                batch_first=True
            ) for _ in range(custom_layers)]
        )

    def forward(self, x):
        return self.layers(x)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


class HebrewDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        with open(data_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        return self.tokenizer(line, truncation=True, return_tensors='pt')


def collate_fn(batch, pad_token_id):
    input_ids = [item['input_ids'].squeeze() for item in batch]
    attention_mask = [item['attention_mask'].squeeze() for item in batch]

    # Pad sequences with the translator's padding token
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return input_ids, attention_mask


def load_data(data_path, tokenizer, batch_size):
    dataset = HebrewDataset(data_path, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
    )


def evaluate_batch(custom_output, llm_first_layer_output):
    mse = nn.MSELoss()(custom_output, llm_first_layer_output)
    cos_sim = cosine_similarity(custom_output.view(-1), llm_first_layer_output.view(-1), dim=0)
    return mse.item(), cos_sim.item()


def train_custom_layer1(translator_model, custom_layer1, llm_model, train_dataloader, translator_tokenizer,
                        llm_tokenizer, num_epochs, learning_rate, device, logger, eval_every=100):
    custom_layer1.to(device)
    translator_model.to(device)
    llm_model.to(device)
    optimizer = optim.Adam(custom_layer1.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Wrap the LLM with our modified version
    modified_llm = ModifiedLLM(llm_model).to(device)

    logger.info("Starting training of CustomLayer1")

    total_steps = len(train_dataloader)
    for epoch in range(num_epochs):
        custom_layer1.train()
        total_loss = 0
        for step, (batch_x, attention_mask) in enumerate(train_dataloader):
            batch_x, attention_mask = batch_x.to(device), attention_mask.to(device)

            with torch.no_grad():
                translator_model.config.output_hidden_states = True
                # Hebrew to English translation
                translated_output = translator_model.generate(
                    input_ids=batch_x,
                    attention_mask=attention_mask,
                    max_length=128,
                )

                # Decode the translated output using the translator's tokenizer
                translated_text = translator_tokenizer.batch_decode(translated_output, skip_special_tokens=True)

                # Tokenize the translated text with the LLM's tokenizer
                llm_inputs = llm_tokenizer(translated_text,
                                           return_tensors="pt",
                                           padding=True,
                                           truncation=True).to(device)

                # Get the outputs and the hidden states after embeddings
                llm_first_layer_output = modified_llm(**llm_inputs)

                # Tokenize the decoder input sentences
                decoder_inputs = translator_tokenizer(translated_text, return_tensors="pt", padding=True,
                                                      truncation=True,
                                                      add_special_tokens=False)

                # Add the decoder_start_token_id to the beginning of each sequence in decoder_inputs
                decoder_start_token_id = translator_model.config.decoder_start_token_id
                decoder_input_ids = torch.cat(
                    [torch.full((decoder_inputs.input_ids.shape[0], 1), decoder_start_token_id, dtype=torch.long),
                     decoder_inputs.input_ids], dim=1)

                # Create attention masks for decoder inputs
                decoder_attention_mask = torch.cat(
                    [torch.ones((decoder_inputs.attention_mask.shape[0], 1), dtype=torch.long),
                     decoder_inputs.attention_mask], dim=1)

                # Get the translator model's last hidden state
                translator_last_hidden = translator_model(
                    input_ids=batch_x,
                    decoder_input_ids=decoder_input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask
                )

                translator_last_hidden = translator_last_hidden.decoder_hidden_states[-1]

            optimizer.zero_grad()
            custom_output = custom_layer1(translator_last_hidden)

            loss = criterion(custom_output, llm_first_layer_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Evaluate every X batches
            if (step + 1) % eval_every == 0:
                mse, cos_sim = evaluate_batch(custom_output, llm_first_layer_output)
                logger.info(f"Epoch {epoch + 1}, Step {step + 1}: MSE = {mse:.4f}, Cosine Similarity = {cos_sim:.4f}")

            # Print progress bar
            print_progress_bar(step + 1, total_steps, epoch + 1, num_epochs, prefix='Progress:', suffix='Complete',
                               length=50)

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    logger.info("Training of CustomLayer1 completed")
    return custom_layer1


def main(args):
    # Setup logger
    log_dir = os.path.join(os.path.dirname(args.save_path), 'logs')
    logger = setup_logger(log_dir)

    # Device selection logic
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    logger.info("Starting the training process")
    logger.info(f"Arguments: {args}")

    # Load the translator model
    translator_model = MarianMTModel.from_pretrained(args.translator_model_name)
    translator_tokenizer = MarianTokenizer.from_pretrained(args.translator_model_name)

    # Load the LLM
    llm_model = OPTForCausalLM.from_pretrained(args.llm_model_name)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name)

    # Load model configurations
    translator_config, llm_config = translator_model.config, llm_model.config

    # Create CustomLayer1
    custom_layer1 = CustomLayer1(
        he_en_hidden_size=translator_config.d_model,
        llm_hidden_size=llm_config.hidden_size,
        num_attention_heads=llm_config.num_attention_heads,
        ffn_dim=llm_config.ffn_dim if hasattr(llm_config, 'ffn_dim') else llm_config.hidden_size * 4,
        dropout=llm_config.dropout,
        custom_layers=args.custom_layers
    )

    logger.info(f"CustomLayer1 architecture:\n{custom_layer1}")

    # Load data
    train_dataloader = load_data(args.data_path, translator_tokenizer, args.batch_size)

    # Train CustomLayer1
    trained_custom_layer1 = train_custom_layer1(
        translator_model, custom_layer1, llm_model, train_dataloader,
        translator_tokenizer, llm_tokenizer, args.num_epochs, args.learning_rate, device, logger
    )

    # Save the trained model
    save_model(trained_custom_layer1, args.save_path)
    logger.info(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CustomLayer1")
    parser.add_argument("--translator_model_name", type=str, default="Helsinki-NLP/opus-mt-tc-big-he-en",
                        help="Name or path of the Hebrew-English translator model")
    parser.add_argument("--llm_model_name", type=str, default="facebook/opt-350m", help="Name or path of the LLM")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the training data")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained CustomLayer1")
    parser.add_argument("--custom_layers", type=int, default=3, help="Number of custom layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help="Device to use for training (auto, cuda, or cpu)")

    args = parser.parse_args()
    main(args)
