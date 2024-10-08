import argparse
import math
from datetime import datetime
import logging
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, OPTForCausalLM
from torch.utils.data import Dataset, DataLoader


def setup_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")

    # Create a logger
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


class FactorizedEmbedding(nn.Module):
    def __init__(self, hidden_size, vocab_size, bottleneck_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.out_proj = nn.Linear(bottleneck_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.dense(x)
        return self.out_proj(x)


class CustomLLM(nn.Module):
    def __init__(self, he_en_model, en_he_model, llm_model, vocab_size, bottleneck_size):
        super().__init__()

        # Hebrew-English components
        self.he_en_embeddings = he_en_model.shared
        self.he_en_encoder = he_en_model.encoder
        self.he_en_decoder = he_en_model.decoder

        # First custom layer
        self.custom_layer1 = nn.Sequential(
            nn.Linear(he_en_model.config.hidden_size, llm_model.config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(llm_model.config.hidden_size)
        )

        # LLM layers (main body of the model)
        self.main_layers = llm_model.model.decoder.layers

        # Second custom layer
        self.custom_layer2 = nn.Sequential(
            nn.Linear(llm_model.config.hidden_size, en_he_model.config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(en_he_model.config.hidden_size)
        )

        # English-Hebrew components
        self.en_he_encoder = en_he_model.encoder
        self.en_he_decoder_layers = en_he_model.decoder.layers

        # Factorized output projection
        self.output_projection = FactorizedEmbedding(
            en_he_model.config.hidden_size,
            vocab_size,
            bottleneck_size
        )

        # Store the start token ID for the HE-EN decoder
        self.he_en_decoder_start_token_id = he_en_model.config.decoder_start_token_id

        # Freeze layers
        self._freeze_layers()

    def _freeze_layers(self):
        # Helper function to freeze all parameters in a module
        def freeze_module(module):
            for param in module.parameters():
                param.requires_grad = False

        # Freeze Hebrew-English components
        freeze_module(self.he_en_embeddings)
        freeze_module(self.he_en_encoder)
        freeze_module(self.he_en_decoder)

        # Freeze LLM layers
        for layer in self.main_layers:
            freeze_module(layer)

        # Freeze English-Hebrew components
        freeze_module(self.en_he_encoder)
        for layer in self.en_he_decoder_layers:
            freeze_module(layer)

        # Ensure custom layers are trainable
        for param in self.custom_layer1.parameters():
            param.requires_grad = True
        for param in self.custom_layer2.parameters():
            param.requires_grad = True

        # Ensure output projection is trainable
        for param in self.output_projection.parameters():
            param.requires_grad = True

    def forward(self, input_ids, en_target_ids, he_target_ids, attention_mask=None):
        # Ensure input_ids is of type Long
        input_ids = input_ids.long()

        # Ensure input_ids is of type Long
        en_target_ids = en_target_ids.long()
        he_target_ids = he_target_ids.long()

        # 1. Hebrew-English encoding
        encoder_output = self.he_en_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 2. Hebrew-English decoding with teacher forcing
        # Shift English target_ids right, adding start token at the beginning
        he_en_decoder_input_ids = torch.cat([
            torch.full((en_target_ids.shape[0], 1), self.he_en_decoder_start_token_id, device=en_target_ids.device),
            en_target_ids[:, :-1]
        ], dim=1)

        he_en_decoder_output = self.he_en_decoder(
            input_ids=he_en_decoder_input_ids,
            encoder_hidden_states=encoder_output,
            attention_mask=attention_mask
        ).last_hidden_state

        # 3. First custom layer
        x = self.custom_layer1(he_en_decoder_output)

        # 4. LLM processing
        for layer in self.llm_layers:
            x = layer(hidden_states=x, attention_mask=attention_mask)[0]

        # 5. Second custom layer
        x = self.custom_layer2(x)

        # 6. English-Hebrew encoding
        en_he_encoder_output = self.en_he_encoder(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state

        # 7. English-Hebrew decoding with teacher forcing
        # Shift Hebrew target_ids right, adding start token at the beginning
        en_he_decoder_input_ids = torch.cat([
            torch.full((he_target_ids.shape[0], 1), self.en_he_decoder.config.decoder_start_token_id,
                       device=he_target_ids.device),
            he_target_ids[:, :-1]
        ], dim=1)

        final_output = self.en_he_decoder(
            input_ids=en_he_decoder_input_ids,
            encoder_hidden_states=en_he_encoder_output,
            attention_mask=attention_mask
        ).last_hidden_state

        # 8. Final projection
        logits = self.output_projection(final_output)

        return logits


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, eval_split=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()

        self.tokenized_data = tokenizer.encode(self.data)

        # Split data into train and eval
        split_point = int(len(self.tokenized_data) * (1 - eval_split))
        self.train_data = self.tokenized_data[:split_point]
        self.eval_data = self.tokenized_data[split_point:]

    def __len__(self):
        return len(self.train_data) - self.max_length

    def __getitem__(self, idx):
        chunk = self.train_data[idx:idx + self.max_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def get_eval_data(self):
        eval_dataset = []
        for i in range(0, len(self.eval_data) - self.max_length, self.max_length):
            chunk = self.eval_data[i:i + self.max_length + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            eval_dataset.append((x, y))
        return eval_dataset


def print_progress_bar(iteration, total, epoch, num_epochs, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create a terminal progress bar with step and epoch information.
    :param iteration: current iteration (int)
    :param total: total iterations (int)
    :param epoch: current epoch (int)
    :param num_epochs: total number of epochs (int)
    :param prefix: prefix string (str)
    :param suffix: suffix string (str)
    :param decimals: positive number of decimals in percent complete (int)
    :param length: character length of bar (int)
    :param fill: bar fill character (str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    step_info = f"Step {iteration}/{total}"
    epoch_info = f"Epoch {epoch}/{num_epochs}"
    print(f'\r{prefix} |{bar}| {percent}% {step_info} {epoch_info} {suffix}', end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def print_model_info(model, logger):
    logger.info("\nModel Architecture:")
    logger.info(str(model))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    logger.info("\nLayer-wise parameter count:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        trainable_param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"{name}: Total params = {param_count:,}, Trainable params = {trainable_param_count:,}")

    # Optional: If you want to log the model summary to a file
    with open('model_summary.txt', 'w') as f:
        f.write(str(model))
    logger.info("Model summary has been saved to 'model_summary.txt'")


def evaluate_batch(logits, targets):
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    # Accuracy
    pred = logits.argmax(dim=-1)
    correct = (pred == targets).float().sum()
    total = targets.numel()
    accuracy = correct / total

    # Perplexity
    perplexity = math.exp(loss.item())

    return loss.item(), accuracy.item(), perplexity


# Helper function for evaluation
def evaluate_full(model, dataloader, he_en_model, device, dataset_name):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Generate English translation
            en_translation = he_en_model.generate(batch_x)

            logits = model(batch_x, en_translation, batch_y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), reduction='sum')
            total_loss += loss.item()

            # Accuracy
            pred = logits.argmax(dim=-1)
            total_correct += (pred == batch_y).sum().item()
            total_tokens += batch_y.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "dataset": dataset_name,
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity
    }


def train_llm(model, dataset, he_en_model, tokenizer, num_epochs=5, batch_size=8, learning_rate=5e-5, device='cuda',
              log_dir='logs', save_dir='model_checkpoints', display_interval=100):
    logger = setup_logger(log_dir)
    logger.info(f"Using device: {device}")
    model.to(device)
    he_en_model.to(device)

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logger.info("Model Architecture:")
    print_model_info(model, logger)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = dataset.get_eval_data()
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Training dataset size: {len(dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    logger.info(f"Number of training batches per epoch: {len(train_dataloader)}")

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'custom_layer' in n or n.startswith('output_projection')],
         'lr': learning_rate},
        {'params': [p for n, p in model.named_parameters() if
                    'custom_layer' not in n and not n.startswith('output_projection')], 'lr': learning_rate / 10}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    logger.info("Starting training...")
    model.train()
    global_step = 0
    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Generate English translation
            with torch.no_grad():
                en_translation = he_en_model.generate(batch_x)

            optimizer.zero_grad()

            logits = model(batch_x, en_translation, batch_y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), ignore_index=tokenizer.pad_token_id)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # Evaluate on current batch every 10 steps
            if global_step % 10 == 0:
                batch_loss, batch_accuracy, batch_perplexity = evaluate_batch(logits, batch_y)
                logger.info(f"Step {global_step}, Batch metrics:")
                logger.info(f"  Loss: {batch_loss:.4f}")
                logger.info(f"  Accuracy: {batch_accuracy:.4f}")
                logger.info(f"  Perplexity: {batch_perplexity:.4f}")

            # Display prediction vs actual label every display_interval steps
            if global_step % display_interval == 0:
                with torch.no_grad():
                    # Get the last sequence in the batch for display
                    input_sequence = batch_x[-1]
                    target_sequence = batch_y[-1]
                    en_sequence = en_translation[-1]

                    # Get the predicted token ids
                    predicted_ids = torch.argmax(logits[-1], dim=-1)

                    # Decode the sequences
                    input_text = tokenizer.decode(input_sequence, skip_special_tokens=True)
                    predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                    target_text = tokenizer.decode(target_sequence, skip_special_tokens=True)
                    en_text = tokenizer.decode(en_sequence, skip_special_tokens=True)

                    logger.info(f"\nStep {global_step}, Prediction vs Actual:")
                    logger.info(f"Input (Hebrew): {input_text}")
                    logger.info(f"Intermediate (English): {en_text}")
                    logger.info(f"Predicted (Hebrew): {predicted_text}")
                    logger.info(f"Actual (Hebrew): {target_text}")

            # Update progress bar
            print_progress_bar(i + 1, len(train_dataloader), epoch + 1, num_epochs,
                               prefix='Training:', suffix=f'Loss: {loss.item():.4f}', length=30)

            # Evaluate on full datasets every half epoch
            if (i + 1) % (len(train_dataloader) // 2) == 0:
                model.eval()
                eval_metrics = evaluate_full(model, eval_dataloader, he_en_model, device, "Evaluation")

                logger.info(f"Full dataset metrics at epoch {epoch + 1}, step {i + 1}:")
                logger.info(f"  {eval_metrics['dataset']} dataset:")
                logger.info(f"    Loss: {eval_metrics['loss']:.4f}")
                logger.info(f"    Accuracy: {eval_metrics['accuracy']:.4f}")
                logger.info(f"    Perplexity: {eval_metrics['perplexity']:.4f}")

                model.train()  # Set the model back to training mode

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.2e}")

        # Save model after each epoch
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

        # Save best model based on evaluation loss
        if eval_metrics['loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['loss']
            best_model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_eval_loss,
            }, best_model_path)
            logger.info(f"Best model saved to {best_model_path}")

        scheduler.step()

    logger.info("Training completed!")


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

    # Create custom LLM
    custom_llm = CustomLLM(he_en_model, en_he_model, llm_model, len(tokenizer), args.bottleneck_size)

    # Train the model
    train_llm(custom_llm, dataset, he_en_model, tokenizer,
              num_epochs=args.num_epochs,
              batch_size=args.batch_size,
              learning_rate=args.learning_rate,
              device=args.device,
              log_dir=args.log_dir,
              save_dir=args.save_dir)


if __name__ == "__main__":
    main()
