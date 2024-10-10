import torch
import torch.nn.functional as F
import math
import os
from utils import print_progress_bar, print_model_info, setup_logger


class Trainer:
    def __init__(self, model, he_en_model, tokenizer, device, log_dir, save_dir, checkpoint):
        self.scheduler_state_dict = None
        self.optimizer_state_dict = None
        self.model = model.to(device)
        self.he_en_model = he_en_model.to(device)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.start_token_id = tokenizer.eos_token_id

        self.device = device
        self.logger = setup_logger(log_dir)
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')

        self.logger.info(f"Using device: {device}")
        self.logger.info("Model Architecture:")
        print_model_info(model, self.logger)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.start_epoch = 0
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint_path):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_eval_loss = checkpoint['loss']
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        self.optimizer_state_dict, self.scheduler_state_dict = checkpoint['optimizer_state_dict'], checkpoint['scheduler_state_dict']

    def evaluate_batch(self, logits, targets):
        # Create a mask to ignore both start and pad tokens
        ignore_mask = (targets != self.start_token_id) & (targets != self.pad_token_id)

        # Apply the mask to both logits and targets
        filtered_logits = logits.view(-1, logits.size(-1))[ignore_mask.view(-1)]
        filtered_targets = targets.view(-1)[ignore_mask.view(-1)]

        # Calculate loss using the filtered logits and targets
        loss = F.cross_entropy(filtered_logits, filtered_targets)

        # Calculate accuracy
        pred = logits.argmax(dim=-1)
        correct = ((pred == targets) & ignore_mask).float().sum()
        total = ignore_mask.float().sum()
        accuracy = correct / total if total > 0 else 0.0

        # Calculate perplexity
        perplexity = math.exp(loss.item())

        return loss.item(), accuracy.item(), perplexity

    def evaluate_full(self, dataloader, dataset_name):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0, 0, 0
        with torch.no_grad():
            for batch in dataloader:
                # Unpack the batch correctly
                batch_x, batch_y, he_attention_mask = batch
                batch_x, batch_y, he_attention_mask = batch_x.to(self.device), batch_y.to(
                    self.device), he_attention_mask.to(self.device)

                en_translation = self.he_en_model.generate(batch_x, attention_mask=he_attention_mask)

                # Create new attention mask for the English translation
                en_attention_mask = (en_translation != self.tokenizer.pad_token_id).float()

                # Create OPT attention mask for the LLM
                llm_attention_mask = create_opt_attention_mask(en_translation, self.tokenizer.pad_token_id)

                logits = self.model(
                    batch_x, en_translation,
                    he_attention_mask=he_attention_mask,
                    en_attention_mask=en_attention_mask,
                    llm_attention_mask=llm_attention_mask
                )

                # Create a mask to ignore both start and pad tokens
                ignore_mask = (batch_y != self.start_token_id) & (batch_y != self.pad_token_id)

                # Apply the mask to both logits and targets
                filtered_logits = logits.view(-1, logits.size(-1))[ignore_mask.view(-1)]
                filtered_targets = batch_y.view(-1)[ignore_mask.view(-1)]

                # Calculate loss using the filtered logits and targets
                loss = F.cross_entropy(filtered_logits, filtered_targets, reduction='sum')
                total_loss += loss.item()

                pred = logits.argmax(dim=-1)
                total_correct += ((pred == batch_y) & ignore_mask).sum().item()
                total_tokens += ignore_mask.sum().item()

        avg_loss = total_loss / total_tokens
        accuracy = total_correct / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "dataset": dataset_name,
            "loss": avg_loss,
            "accuracy": accuracy,
            "perplexity": perplexity
        }

    def log_prediction(self, batch_x, batch_y, en_translation, logits, step):
        input_sequence = batch_x[-1]
        target_sequence = batch_y[-1]
        en_sequence = en_translation[-1]
        predicted_ids = torch.argmax(logits[-1], dim=-1)

        input_text = self.tokenizer.decode(input_sequence, skip_special_tokens=False)
        predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=False)
        target_text = self.tokenizer.decode(target_sequence, skip_special_tokens=False)
        en_text = self.tokenizer.decode(en_sequence, skip_special_tokens=False)

        self.logger.info(f"Step {step}, Prediction vs Actual:")
        self.logger.info(f"Input (Hebrew): {input_text}")
        self.logger.info(f"Intermediate (English): {en_text}")
        self.logger.info(f"Predicted (Hebrew): {predicted_text}")
        self.logger.info(f"Actual (Hebrew): {target_text}")

        # Token-wise comparison
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_ids)
        target_tokens = self.tokenizer.convert_ids_to_tokens(target_sequence)

        # Ensure both sequences have the same length for comparison
        max_length = max(len(predicted_tokens), len(target_tokens))
        predicted_tokens = predicted_tokens + [''] * (max_length - len(predicted_tokens))
        target_tokens = target_tokens + [''] * (max_length - len(target_tokens))

        # Create comparison table
        table = "| Index | Predicted Token | Actual Token | Match |\n"
        table += "|-------|-----------------|--------------|-------|\n"
        for i, (pred, target) in enumerate(zip(predicted_tokens, target_tokens)):
            match = "✓" if pred == target else "✗"
            table += f"| {i:5d} | {pred:15s} | {target:12s} | {match:5s} |\n"

        self.logger.info("Token-wise comparison:")
        self.logger.info("\n" + table)

    def save_checkpoint(self, epoch, optimizer, scheduler, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }
        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pt')
            self.logger.info(f"New best model saved to {path}")
        else:
            path = os.path.join(self.save_dir, f'model_epoch_{epoch + 1}.pt')
            self.logger.info(f"Model saved to {path}")
        torch.save(checkpoint, path)

    def train(self, train_dataloader, eval_dataloader, num_epochs, learning_rate, checkpoint, display_interval):
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() if
                        'custom_layer' in n or n.startswith('output_projection')],
             'lr': learning_rate},
            {'params': [p for n, p in self.model.named_parameters() if
                        'custom_layer' not in n and not n.startswith('output_projection')], 'lr': learning_rate / 10}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        if self.start_epoch > 0:
            optimizer_state_dict, scheduler_state_dict = self.optimizer_state_dict, self.scheduler_state_dict
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler.load_state_dict(scheduler_state_dict)

        self.logger.info(f"Starting training from epoch {self.start_epoch}")
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for i, (batch_x, batch_y, he_attention_mask) in enumerate(train_dataloader):
                batch_x, batch_y, he_attention_mask = batch_x.to(self.device), batch_y.to(self.device), he_attention_mask.to(self.device)

                with torch.no_grad():
                    # Generate English translation
                    en_translation = self.he_en_model.generate(batch_x, attention_mask=he_attention_mask)

                    # Create new attention mask for the English translation
                    en_attention_mask = (en_translation != self.tokenizer.pad_token_id).float()

                    # Create OPT attention mask for the LLM
                    llm_attention_mask = create_opt_attention_mask(en_translation, self.tokenizer.pad_token_id)

                optimizer.zero_grad()
                logits = self.model(
                    batch_x, en_translation,
                    he_attention_mask=he_attention_mask,
                    en_attention_mask=en_attention_mask,
                    llm_attention_mask=llm_attention_mask
                )

                # Create a mask to ignore both start and pad tokens
                ignore_mask = (batch_y != self.start_token_id) & (batch_y != self.pad_token_id)

                # Apply the mask to both logits and targets
                filtered_logits = logits.view(-1, logits.size(-1))[ignore_mask.view(-1)]
                filtered_targets = batch_y.view(-1)[ignore_mask.view(-1)]

                # Calculate loss using the filtered logits and targets
                loss = F.cross_entropy(filtered_logits, filtered_targets)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                global_step += 1

                if global_step % 10 == 0:
                    batch_metrics = self.evaluate_batch(logits, batch_y)
                    self.logger.info(f"Step {global_step}, Batch metrics: "
                                     f"Loss: {batch_metrics[0]:.4f}, "
                                     f"Accuracy: {batch_metrics[1]:.4f}, "
                                     f"Perplexity: {batch_metrics[2]:.4f}")

                if global_step % display_interval == 0:
                    self.log_prediction(batch_x, batch_y, en_translation, filtered_logits, global_step)

                print_progress_bar(i + 1, len(train_dataloader), epoch + 1, num_epochs,
                                   prefix='Training:', suffix=f'Loss: {loss.item():.4f}', length=30)

                if (i + 1) % (len(train_dataloader) // 2) == 0:
                    eval_metrics = self.evaluate_full(eval_dataloader, "Evaluation")
                    self.logger.info(f"Full dataset metrics at epoch {epoch + 1}, step {i + 1}:")
                    self.logger.info(f"  {eval_metrics['dataset']} dataset:")
                    self.logger.info(f"    Loss: {eval_metrics['loss']:.4f}")
                    self.logger.info(f"    Accuracy: {eval_metrics['accuracy']:.4f}")
                    self.logger.info(f"    Perplexity: {eval_metrics['perplexity']:.4f}")

                    if eval_metrics['loss'] < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics['loss']
                        self.save_checkpoint(epoch, optimizer, scheduler, self.best_eval_loss, is_best=True)

            avg_loss = total_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")
            self.logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            self.save_checkpoint(epoch, optimizer, scheduler, avg_loss)
            scheduler.step()

        self.logger.info("Training completed!")


def create_opt_attention_mask(input_ids, padding_idx=1):
    """
    Create a causal attention mask for the OPT model.

    Args:
    input_ids (torch.Tensor): Input tensor of shape (batch_size, sequence_length)
    padding_idx (int): The index used for padding, default is 1 for OPT models

    Returns:
    torch.Tensor: Attention mask of shape (batch_size, 1, sequence_length, sequence_length)
    """
    batch_size, seq_length = input_ids.size()

    # Create a mask for padding tokens
    padding_mask = (input_ids != padding_idx).long()

    # Create a causal mask
    causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=input_ids.device))

    # Combine padding mask and causal mask
    attention_mask = padding_mask.unsqueeze(1).unsqueeze(2) * causal_mask.unsqueeze(0)

    # OPT models typically expect the attention mask to have values 0 for attended positions and -10000 for masked positions
    attention_mask = attention_mask.float().masked_fill(attention_mask == 0, float('-inf')).masked_fill(
        attention_mask == 1, 0.0)

    return attention_mask


def train_llm(model, train_dataloader, eval_dataloader, he_en_model, tokenizer, num_epochs, learning_rate, device,
              log_dir, save_dir, checkpoint, display_interval=100):
    trainer = Trainer(model, he_en_model, tokenizer, device, log_dir, save_dir, checkpoint)
    trainer.train(train_dataloader, eval_dataloader, num_epochs, learning_rate, checkpoint, display_interval)
