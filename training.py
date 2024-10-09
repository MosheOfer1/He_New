import torch
import torch.nn.functional as F
import math
import os
from utils import print_progress_bar, print_model_info, setup_logger


class Trainer:
    def __init__(self, model, he_en_model, tokenizer, device, log_dir, save_dir):
        self.model = model.to(device)
        self.he_en_model = he_en_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.logger = setup_logger(log_dir)
        self.save_dir = save_dir
        self.best_eval_loss = float('inf')

        self.logger.info(f"Using device: {device}")
        self.logger.info("Model Architecture:")
        print_model_info(model, self.logger)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def evaluate_batch(self, logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        pred = logits.argmax(dim=-1)
        correct = (pred == targets).float().sum()
        total = targets.numel()
        accuracy = correct / total
        perplexity = math.exp(loss.item())
        return loss.item(), accuracy.item(), perplexity

    def evaluate_full(self, dataloader, dataset_name):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                en_translation = self.he_en_model.generate(batch_x)
                logits = self.model(batch_x, en_translation, batch_y)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1), reduction='sum')
                total_loss += loss.item()
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

    def log_prediction(self, batch_x, batch_y, en_translation, logits, step):
        input_sequence = batch_x[-1]
        target_sequence = batch_y[-1]
        en_sequence = en_translation[-1]
        predicted_ids = torch.argmax(logits[-1], dim=-1)

        input_text = self.tokenizer.decode(input_sequence, skip_special_tokens=True)
        predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
        target_text = self.tokenizer.decode(target_sequence, skip_special_tokens=True)
        en_text = self.tokenizer.decode(en_sequence, skip_special_tokens=True)

        self.logger.info(f"\nStep {step}, Prediction vs Actual:")
        self.logger.info(f"Input (Hebrew): {input_text}")
        self.logger.info(f"Intermediate (English): {en_text}")
        self.logger.info(f"Predicted (Hebrew): {predicted_text}")
        self.logger.info(f"Actual (Hebrew): {target_text}")

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

    def train(self, train_dataloader, eval_dataloader, num_epochs, learning_rate, display_interval=100):
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in self.model.named_parameters() if
                        'custom_layer' in n or n.startswith('output_projection')],
             'lr': learning_rate},
            {'params': [p for n, p in self.model.named_parameters() if
                        'custom_layer' not in n and not n.startswith('output_projection')], 'lr': learning_rate / 10}
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.logger.info("Starting training...")
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
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1),
                                       ignore_index=self.tokenizer.pad_token_id)
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
                    self.log_prediction(batch_x, batch_y, en_translation, logits, global_step)

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
              log_dir, save_dir, display_interval=100):
    trainer = Trainer(model, he_en_model, tokenizer, device, log_dir, save_dir)
    trainer.train(train_dataloader, eval_dataloader, num_epochs, learning_rate, display_interval)
