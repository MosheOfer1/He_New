import copy

from torch import nn
from transformers import MarianConfig, MarianMTModel, MarianTokenizer, AutoModelForCausalLM, AutoTokenizer

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedModel
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from auto_encoder import DimensionAlignmentAutoencoder
from utils import print_model_info, setup_logger


class HebrewDataset(Dataset):
    def __init__(self, hebrew_sentences: List[str],
                 he_en_tokenizer,
                 max_length: int = 512):
        self.hebrew_sentences = hebrew_sentences
        self.he_en_tokenizer = he_en_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hebrew_sentences)

    def __getitem__(self, idx):
        hebrew_text = self.hebrew_sentences[idx]

        # Tokenize Hebrew input without padding
        hebrew_tokens = self.he_en_tokenizer(
            hebrew_text,
            max_length=self.max_length,
            truncation=True,
        )

        return {
            'hebrew_input_ids': torch.tensor(hebrew_tokens['input_ids']),
            'hebrew_attention_mask': torch.tensor(hebrew_tokens['attention_mask']),
        }


def collate_fn(batch):
    # Separate the different components of the batch
    hebrew_input_ids = [item['hebrew_input_ids'] for item in batch]
    hebrew_attention_mask = [item['hebrew_attention_mask'] for item in batch]

    # Pad sequences
    hebrew_input_ids_padded = pad_sequence(hebrew_input_ids, batch_first=True, padding_value=0)
    hebrew_attention_mask_padded = pad_sequence(hebrew_attention_mask, batch_first=True, padding_value=0)

    return {
        'hebrew_input_ids': hebrew_input_ids_padded,
        'hebrew_attention_mask': hebrew_attention_mask_padded,
    }


class CustomTrainer:
    def __init__(
            self,
            model: nn.Module,
            he_en_model: PreTrainedModel,
            llm_model: PreTrainedModel,
            he_en_tokenizer,
            llm_tokenizer,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            device = 'cuda' if torch.cuda.is_available() else 'cpu',
            batch_size: int = 8,
            learning_rate: float = 2e-5,
            max_length: int = 512,
    ):
        self.model = model.to(device)
        self.he_en_model = he_en_model.to(device)
        self.llm_model = llm_model.to(device)
        self.he_en_tokenizer = he_en_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Set models to appropriate modes
        self.he_en_model.eval()
        self.llm_model.eval()

        # Create data loaders with custom collate function
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_fn
            )

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.logger = setup_logger('logs')
        print_model_info(model, self.logger)


    @torch.no_grad()
    def translate_and_get_states(self,
                                 input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor) -> tuple[Any, Any]:

        translation_outputs = self.he_en_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_length=self.max_length,
            num_beams=1,
            output_scores=True,
        )

        # Rest of the tokenization and labeling process
        translated_ids = translation_outputs.sequences
        translated_text = self.he_en_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        translated_text_with_bos = ["<s>" + text for text in translated_text]

        llm_tokens = self.llm_tokenizer(
            translated_text_with_bos,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)

        labels = self.llm_tokenizer(
            translated_text,
            max_length=self.max_length - 1,
            truncation=True,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(self.device)

        # Handle labels and EOS token
        labels_input_ids = labels.input_ids
        batch_size, seq_len = labels_input_ids.shape
        new_labels = torch.full((batch_size, seq_len + 1),
                                self.llm_tokenizer.pad_token_id,
                                dtype=labels_input_ids.dtype,
                                device=labels_input_ids.device)
        new_attention_mask = torch.zeros((batch_size, seq_len + 1),
                                         dtype=labels.attention_mask.dtype,
                                         device=labels.attention_mask.device)

        new_labels[:, :seq_len] = labels_input_ids
        new_attention_mask[:, :seq_len] = labels.attention_mask

        pad_token_id = self.llm_tokenizer.pad_token_id
        eos_token_id = self.llm_tokenizer.eos_token_id

        for i in range(batch_size):
            pad_start = (labels_input_ids[i] == pad_token_id).nonzero(as_tuple=True)[0]
            if len(pad_start) > 0:
                pad_start = pad_start[0]
                new_labels[i, pad_start] = eos_token_id
                new_attention_mask[i, pad_start] = 1
            else:
                new_labels[i, -1] = eos_token_id
                new_attention_mask[i, -1] = 1

        labels.input_ids = new_labels
        labels.attention_mask = new_attention_mask

        return llm_tokens, labels

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, List[str]]:
        self.model.train()
        self.optimizer.zero_grad()

        # Get Hebrew input tensors
        hebrew_input_ids = batch['hebrew_input_ids'].to(self.device)
        hebrew_attention_mask = batch['hebrew_attention_mask'].to(self.device)

        # Get encoder states, decoder states, and translated text
        llm_tokens, labels = self.translate_and_get_states(
            hebrew_input_ids,
            hebrew_attention_mask
        )

        # Get LLM embeddings for the translated text
        llm_embeddings = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)

        # Forward pass through main model without labels
        outputs = self.model(
            input_ids=hebrew_input_ids,
            decoder_inputs_embeds=llm_embeddings,
            attention_mask=hebrew_attention_mask,
            decoder_attention_mask=llm_tokens.attention_mask,
        )

        # Get logits from output and vocab size
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        vocab_size = logits.shape[-1]

        # Clamp labels to ensure they're within vocab size
        target = labels.input_ids.clamp(0, vocab_size - 1)

        # Reshape logits and labels
        logits = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        target = target.view(-1)  # (batch_size * seq_len)

        # Calculate cross entropy loss with padding token ignored
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.llm_tokenizer.pad_token_id)
        loss = loss_fct(logits, target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.eval()

        # Get Hebrew input tensors
        hebrew_input_ids = batch['hebrew_input_ids'].to(self.device)
        hebrew_attention_mask = batch['hebrew_attention_mask'].to(self.device)

        # tokens
        llm_tokens, labels = self.translate_and_get_states(
            hebrew_input_ids,
            hebrew_attention_mask
        )

        # Get LLM embeddings for the translated text
        llm_embeddings = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)

        # Forward pass through main model
        outputs = self.model(
            input_ids=hebrew_input_ids,
            decoder_inputs_embeds=llm_embeddings,
            attention_mask=hebrew_attention_mask,
            decoder_attention_mask=llm_tokens.attention_mask,
        )

        # Get logits and vocab size
        logits = outputs.logits
        vocab_size = logits.shape[-1]

        # Clamp labels
        target = labels.input_ids.clamp(0, vocab_size - 1)

        # Reshape for loss calculation
        logits = logits.view(-1, vocab_size)
        target = target.view(-1)

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.llm_tokenizer.pad_token_id)
        loss = loss_fct(logits, target)

        return loss.item()

    def train(self, num_epochs: int, validate_every: int = 1):
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            # Training loop
            epoch_losses = []
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_losses.append(loss)

                progress_bar.set_postfix({
                    'loss': f"{np.mean(epoch_losses[-100:]):.4f}",
                    'heb_len': f"{np.mean(batch['hebrew_attention_mask'].sum(1).tolist()):.1f}",
                })

            avg_train_loss = np.mean(epoch_losses)
            train_losses.append(avg_train_loss)

            # Validation loop
            if self.val_loader is not None and (epoch + 1) % validate_every == 0:
                val_epoch_losses = []
                for batch in tqdm(self.val_loader, desc='Validating'):
                    val_loss = self.validate_step(batch)
                    val_epoch_losses.append(val_loss)

                avg_val_loss = np.mean(val_epoch_losses)
                val_losses.append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    # Add model saving here if needed

                print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
            else:
                print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}')

        return train_losses, val_losses


class Transformer1(nn.Module):
    def __init__(self,
                 he_en_model,
                 llm_model,
                 align_he_en: DimensionAlignmentAutoencoder = None,
                 ):
        super().__init__()

        # Create a new decoder config with matching dimensions
        config = MarianConfig(
            d_model=llm_model.config.hidden_size,
            encoder_attention_heads=he_en_model.config.encoder_attention_heads,
            encoder_ffn_dim=he_en_model.config.encoder_ffn_dim,
            encoder_layers=he_en_model.config.encoder_layers,
            decoder_attention_heads=llm_model.config.num_attention_heads,
            decoder_ffn_dim=he_en_model.config.decoder_ffn_dim,
            decoder_layers=he_en_model.config.decoder_layers,
            max_length=he_en_model.config.max_length,
            vocab_size=llm_model.config.vocab_size,
            scale_embedding=he_en_model.config.scale_embedding,
            pad_token_id=llm_model.config.pad_token_id,
            eos_token_id=llm_model.config.eos_token_id,
            decoder_start_token_id=llm_model.config.decoder_start_token_id,
            share_encoder_decoder_embeddings=True
        )

        self.main_model = MarianMTModel(config=config)
        self.main_model.set_input_embeddings(copy.deepcopy(he_en_model.get_input_embeddings()))

        # # Dimension alignment layer
        # if align_he_en is None:
        #     self.align_he_en = DimensionAlignmentAutoencoder(
        #         input_dim=he_en_model.config.d_model,
        #         target_dim=llm_model.config.hidden_size
        #     ).encoder
        # else:
        #     self.align_he_en = align_he_en.encoder

    def forward(
            self,
            input_ids,
            decoder_inputs_embeds,
            attention_mask,
            decoder_attention_mask,
    ):
        # inputs_embeds = self.align_he_en(inputs_embeds)
        outputs = self.main_model(
            input_ids=input_ids,
            decoder_inputs_embeds=decoder_inputs_embeds,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        return outputs

    # Add this method to the Transformer1 class
    def generate(self,
                 hebrew_text: str,
                 he_en_tokenizer,
                 llm_tokenizer,
                 he_en_model,
                 llm_model,
                 max_length: int = 512,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 max_new_tokens: int = 128,
                 temperature: float = 1.0,
                 top_p: float = 0.9) -> dict:
        """
        Generate output for a Hebrew input sentence.
        Returns both the English translation and the LLM output.
        """
        self.eval()
        with torch.no_grad():
            # Tokenize Hebrew input
            hebrew_tokens = he_en_tokenizer(
                hebrew_text,
                max_length=max_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Get translation and decoder states
            translation_outputs = he_en_model.generate(
                input_ids=hebrew_tokens['input_ids'],
                attention_mask=hebrew_tokens['attention_mask'],
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_length=max_length,
                num_beams=1,
                output_scores=True,
            )

            # Get decoder hidden states
            decoder_hidden_states = [step[-1] for step in translation_outputs.decoder_hidden_states]
            hidden_states = torch.stack(decoder_hidden_states, dim=0)

            # Reshape hidden states
            batch_size = 1
            hidden_states = hidden_states.view(-1, batch_size, 1, hidden_states.shape[-1])
            decoder_states = hidden_states[:, 0, 0].unsqueeze(0)  # Add batch dimension

            # Create attention mask for decoder states
            decoder_attention_mask = torch.ones(
                (1, decoder_states.size(1)),
                dtype=torch.long,
                device=device
            )

            # Get English translation
            translated_ids = translation_outputs.sequences
            translated_text = he_en_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]

            # Align dimensions for decoder input
            decoder_states = self.align_he_en(decoder_states)

            # Initialize generation with proper handling of special tokens
            generated_ids = []
            current_token = llm_tokenizer.encode("<|im_start|>", add_special_tokens=False, return_tensors='pt').to(
                device)

            for _ in range(max_new_tokens):
                # Get embeddings for current token
                current_embeds = llm_model.get_input_embeddings()(current_token)

                try:
                    outputs = self.main_model(
                        inputs_embeds=decoder_states,
                        decoder_inputs_embeds=current_embeds,
                        attention_mask=decoder_attention_mask,
                        decoder_attention_mask=torch.ones_like(current_token),
                    )

                    # Get next token logits and apply temperature
                    next_token_logits = outputs.logits[:, -1, :] / temperature

                    # Apply softmax first, then filter
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Filter vocabulary to only valid tokens
                    valid_tokens_mask = torch.ones_like(probs, dtype=torch.bool)
                    valid_tokens_mask[:, llm_tokenizer.vocab_size:] = False  # Mask out invalid token ids
                    probs = probs * valid_tokens_mask.float()

                    # Renormalize probabilities
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                    # Apply top-p filtering
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Apply the filtering
                    # sorted_probs[sorted_indices_to_remove] = 0
                    # probs.scatter_(1, sorted_indices, sorted_probs)

                    # Sample from the filtered distribution
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Break if we hit the EOS token or max length
                    if next_token.item() == llm_tokenizer.eos_token_id:
                        break

                    # Append token and continue
                    generated_ids.append(next_token.item())
                    current_token = torch.cat([current_token, next_token], dim=1)

                except RuntimeError as e:
                    print(f"Error during generation: {e}")
                    break

            # Decode generated text, handling empty sequences
            if generated_ids:
                generated_text = llm_tokenizer.decode(generated_ids, skip_special_tokens=True)
            else:
                generated_text = ""

            return {
                'hebrew_input': hebrew_text,
                'english_translation': translated_text,
                'llm_output': generated_text
            }


def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """Filter a distribution of logits using nucleus (top-p) filtering"""
    if top_p <= 0.0:
        return logits

    # Clamp logits to prevent numerical instability
    logits = torch.clamp(logits, min=-100, max=100)

    probs = torch.softmax(logits, dim=-1)

    # Remove tokens with 0 probability
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p

    # Keep at least min_tokens_to_keep tokens
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value

    return logits

# Function to present results in a table format
def present_results(results: dict) -> str:
    """
    Present the generation results in a formatted table.
    """
    table = (
        "╔════════════════════╤══════════════════════════════════════════════════════════════╗\n"
        "║ Type               │ Text                                                         ║\n"
        "╠════════════════════╪══════════════════════════════════════════════════════════════╣\n"
        f"║ Hebrew Input      │ {results['hebrew_input']:<56} ║\n"
        f"║ English Translation│ {results['english_translation']:<56} ║\n"
        f"║ LLM Output        │ {results['llm_output']:<56} ║\n"
        "╚════════════════════╧══════════════════════════════════════════════════════════════╝"
    )
    return table


def generate_example(model_path: str, hebrew_text: str, device):
    try:
        # Load models and tokenizers
        he_en_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en").to(device)
        he_en_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-he-en")
        llm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)
        llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

        if not llm_tokenizer.pad_token_id:
            llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id

        # Initialize adapter model
        adapter_model = Transformer1(
            he_en_model=he_en_model,
            llm_model=llm_model,
        ).to(device)

        # Load trained weights with error handling
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            adapter_model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return

        # Generate output
        results = adapter_model.generate(
            hebrew_text=hebrew_text,
            he_en_tokenizer=he_en_tokenizer,
            llm_tokenizer=llm_tokenizer,
            he_en_model=he_en_model,
            llm_model=llm_model,
            temperature=0.2,  # Lower temperature for more stable generation
            top_p=0.95  # Slightly higher top_p for more diversity
        )

        # Print results in table format
        print(present_results(results))

    except Exception as e:
        print(f"Error in generate_example: {e}")