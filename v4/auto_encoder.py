import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os


class DimensionAlignmentAutoencoder(nn.Module):
    def __init__(self, input_dim, target_dim, hidden_factor=2):
        """
        Initialize the autoencoder for dimension alignment.

        Args:
            input_dim: Input dimension (he_en_model.config.d_model)
            target_dim: Target dimension (llm_model.config.hidden_size)
            hidden_factor: Factor to multiply target_dim for the hidden layer size
        """
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim

        # Encoder layers that transform to target dimension
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, target_dim * hidden_factor),
            nn.LayerNorm(target_dim * hidden_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * hidden_factor, target_dim),
            nn.LayerNorm(target_dim),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(target_dim, target_dim * hidden_factor),
            nn.LayerNorm(target_dim * hidden_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(target_dim * hidden_factor, input_dim),
            nn.LayerNorm(input_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        """
        Encode input to target dimension.
        """
        # Handle 3D input
        if len(x.shape) == 3:
            batch_size, seq_len, _ = x.shape
            # Reshape to 2D
            x = x.contiguous().view(-1, self.input_dim)
            # Encode
            encoded = self.encoder(x)
            # Reshape back to 3D
            encoded = encoded.view(batch_size, seq_len, self.target_dim)
            return encoded

        # Handle 2D input
        return self.encoder(x)

    def decode(self, z):
        """
        Decode encoded representation back to input dimension.
        """
        # Handle 3D input
        if len(z.shape) == 3:
            batch_size, seq_len, _ = z.shape
            # Reshape to 2D
            z = z.contiguous().view(-1, self.target_dim)
            # Decode
            decoded = self.decoder(z)
            # Reshape back to 3D
            decoded = decoded.view(batch_size, seq_len, self.input_dim)
            return decoded

        # Handle 2D input
        return self.decoder(z)

    def forward(self, x, return_encoded=False):
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_encoded: If True, return both encoded and decoded tensors
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        if return_encoded:
            return encoded, decoded
        return decoded


def debug_shapes(tensor, name):
    """Helper function to debug tensor shapes"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name} shape: {tensor.shape}")
        print(f"{name} size: {tensor.numel()}")
    else:
        print(f"{name} is not a tensor")



class EncoderHiddenStatesDataset(Dataset):
    def __init__(self, sentences, he_en_model, tokenizer1, device='cuda', max_length=128):
        """
        Dataset for getting encoder hidden states on the fly.

        Args:
            sentences: List of Hebrew sentences
            he_en_model: Hebrew-English translation model
            tokenizer1: Hebrew tokenizer
            device: Device to use
            max_length: Maximum sequence length
        """
        self.sentences = sentences
        self.he_en_model = he_en_model
        self.tokenizer1 = tokenizer1
        self.device = device
        self.max_length = max_length

        # Keep encoder in eval mode
        self.he_en_model.eval()

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]

        # Tokenize
        inputs = self.tokenizer1(
            sentence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get encoder hidden states
        with torch.no_grad():
            outputs = self.he_en_model.model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            hidden_states = outputs.last_hidden_state.squeeze(0)  # Remove batch dimension

        return hidden_states

    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences."""
        # Find max length in batch
        max_len = max(x.size(0) for x in batch)

        # Pad sequences to max length
        padded_batch = []
        for hidden_state in batch:
            if hidden_state.size(0) < max_len:
                padding = torch.zeros(
                    (max_len - hidden_state.size(0), hidden_state.size(1)),
                    dtype=hidden_state.dtype,
                    device=hidden_state.device
                )
                hidden_state = torch.cat([hidden_state, padding], dim=0)
            padded_batch.append(hidden_state)

        return torch.stack(padded_batch)


class AutoencoderPreTrainer:
    def __init__(self, autoencoder, he_en_model, tokenizer1, device='cuda'):
        """
        Initialize the autoencoder trainer.

        Args:
            autoencoder: DimensionAlignmentAutoencoder instance
            he_en_model: Hebrew-English translation model
            tokenizer1: Hebrew tokenizer
            device: Device to use for training
        """
        self.autoencoder = autoencoder
        self.he_en_model = he_en_model
        self.tokenizer1 = tokenizer1
        self.device = device

        self.autoencoder.to(device)
        self.he_en_model.to(device)
        self.he_en_model.eval()

        # Initialize logging
        self.train_losses = []
        self.val_losses = []
        self.epoch_train_losses = []
        self.epoch_val_losses = []


    def create_data_loaders(self, sentences, batch_size, validation_split=0.1):
        """Create train and validation datasets and data loaders."""
        # Split sentences into train and validation
        val_size = int(len(sentences) * validation_split)
        train_sentences = sentences[:-val_size]
        val_sentences = sentences[-val_size:]

        # Create datasets
        train_dataset = EncoderHiddenStatesDataset(
            train_sentences,
            self.he_en_model,
            self.tokenizer1,
            self.device
        )
        val_dataset = EncoderHiddenStatesDataset(
            val_sentences,
            self.he_en_model,
            self.tokenizer1,
            self.device
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn
        )

        return train_loader, val_loader


    def save_logs(self, save_dir):
        """Save training logs to files"""
        import json
        import numpy as np

        # Save detailed (batch-level) losses
        np.save(os.path.join(save_dir, 'train_losses.npy'), np.array(self.train_losses))
        np.save(os.path.join(save_dir, 'val_losses.npy'), np.array(self.val_losses))

        # Save epoch-level losses
        logs = {
            'train_losses': self.epoch_train_losses,
            'val_losses': self.epoch_val_losses
        }

        with open(os.path.join(save_dir, 'training_log.json'), 'w') as f:
            json.dump(logs, f, indent=4)

    def plot_losses(self, save_dir):
        """Plot training and validation losses"""
        import matplotlib.pyplot as plt

        # Plot epoch-level losses
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_train_losses, label='Training Loss')
        plt.plot(self.epoch_val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
        plt.close()

        # Plot detailed batch-level training loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Batch-level Training Loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'batch_loss_plot.png'))
        plt.close()

    def train(self,
              sentences,
              num_epochs=100,
              batch_size=32,
              learning_rate=1e-4,
              validation_split=0.1,
              save_dir='checkpoints',
              log_every=100):  # New parameter to control logging frequency

        train_loader, val_loader = self.create_data_loaders(
            sentences, batch_size, validation_split
        )

        optimizer = torch.optim.AdamW(self.autoencoder.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.MSELoss()

        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        no_improve_count = 0

        # Create log file
        log_file = os.path.join(save_dir, 'training.log')

        # Initialize running loss for smooth progress bar display
        running_loss = 0.0
        log_interval_loss = 0.0

        print("Starting training...")
        for epoch in range(num_epochs):
            # Training
            self.autoencoder.train()
            epoch_train_loss = 0
            batch_count = 0

            # Create progress bar with additional stats
            pbar = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f'Epoch {epoch + 1}/{num_epochs}',
                        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            for batch_idx, batch in pbar:
                try:
                    optimizer.zero_grad()

                    # Forward pass
                    output = self.autoencoder(batch)
                    loss = criterion(output, batch)

                    # Store batch loss
                    self.train_losses.append(loss.item())

                    # Update running loss for progress bar
                    running_loss = 0.9 * running_loss + 0.1 * loss.item()  # Exponential moving average
                    log_interval_loss += loss.item()

                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    batch_count += 1

                    # Update progress bar with current loss
                    pbar.set_postfix({
                        'loss': f'{running_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                    # Log every N batches
                    if (batch_idx + 1) % log_every == 0:
                        avg_interval_loss = log_interval_loss / log_every
                        print(f"\nBatch {batch_idx + 1}/{len(train_loader)}, "
                              f"Average Loss: {avg_interval_loss:.4f}, "
                              f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

                        with open(log_file, 'a') as f:
                            f.write(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, "
                                    f"Loss: {avg_interval_loss:.6f}, "
                                    f"LR: {scheduler.get_last_lr()[0]:.2e}\n")

                        # Reset interval loss
                        log_interval_loss = 0.0

                except RuntimeError as e:
                    print(f"\nError in batch {batch_idx}:")
                    print(str(e))
                    continue

            avg_train_loss = epoch_train_loss / batch_count
            self.epoch_train_losses.append(avg_train_loss)

            # Validation with progress bar
            self.autoencoder.eval()
            val_loss = 0
            val_count = 0

            val_pbar = tqdm(val_loader,
                            desc='Validation',
                            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            with torch.no_grad():
                running_val_loss = 0.0
                for batch in val_pbar:
                    try:
                        output = self.autoencoder(batch)
                        batch_loss = criterion(output, batch).item()
                        val_loss += batch_loss
                        self.val_losses.append(batch_loss)
                        val_count += 1

                        # Update running validation loss
                        running_val_loss = 0.9 * running_val_loss + 0.1 * batch_loss
                        val_pbar.set_postfix({'val_loss': f'{running_val_loss:.4f}'})

                    except RuntimeError as e:
                        print(f"Error in validation batch: {str(e)}")
                        continue

            avg_val_loss = val_loss / val_count
            self.epoch_val_losses.append(avg_val_loss)

            # Log epoch results
            log_message = (
                f'\nEpoch {epoch + 1}/{num_epochs}:\n'
                f'Train Loss: {avg_train_loss:.6f}\n'
                f'Val Loss: {avg_val_loss:.6f}\n'
                f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}\n'
            )

            print(log_message)
            with open(log_file, 'a') as f:
                f.write(log_message + '\n')

            # Update learning rate
            scheduler.step(avg_val_loss)

            # Save best model and handle early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_count = 0

                # Save model and logs
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.autoencoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'epoch_train_losses': self.epoch_train_losses,
                    'epoch_val_losses': self.epoch_val_losses,
                }, os.path.join(save_dir, 'best_autoencoder.pt'))

                print(f'Saved new best model with validation loss: {avg_val_loss:.6f}')
            else:
                no_improve_count += 1

            # Save current logs and plots
            self.save_logs(save_dir)
            self.plot_losses(save_dir)

            if no_improve_count >= 10:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        print("Training completed!")
        return best_val_loss