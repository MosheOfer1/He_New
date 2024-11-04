import torch
import torch.nn as nn
from transformers.models.marian.modeling_marian import MarianDecoder, MarianEncoder
import torch.nn.functional as F

from utils import create_opt_attention_mask


class EmbeddingLLM(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.embed_tokens = original_model.model.decoder.embed_tokens
        self.project_in = original_model.model.decoder.project_in
        self.embed_positions = original_model.model.decoder.embed_positions

    def forward(self, input_ids, attention_mask=None):
        # Get the embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.project_in(inputs_embeds)

        # Add positional embeddings
        pos_embeds = self.embed_positions(attention_mask, 0)
        hidden_states = inputs_embeds + pos_embeds

        # Return the hidden states after embeddings
        return hidden_states


class CustomLLM(nn.Module):
    def __init__(self, he_en_model, en_he_model, llm_model):
        super().__init__()

        # Hebrew-English components
        self.he_en_model_encoder = he_en_model.model.encoder

        # First custom transformer layers
        self.custom_embedding = EmbeddingLLM(llm_model)
        self.custom_decoder1 = MarianDecoder(he_en_model.config)
        self.custom_decoder1.set_input_embeddings(None)

        # Linear layer between custom_decoder1 and main_layers
        self.linear1 = nn.Linear(he_en_model.config.d_model, llm_model.config.hidden_size)

        # LLM layers (main body of the model)
        self.main_layers = llm_model.model.decoder.layers

        # Linear layer between main_layers and custom_encoder2
        self.linear2 = nn.Linear(llm_model.config.hidden_size, en_he_model.config.d_model)

        # Second custom transformer layers
        self.custom_encoder2 = MarianEncoder(en_he_model.config, llm_model.model.decoder.embed_tokens)
        self.custom_encoder2.set_input_embeddings(None)

        # English-Hebrew components
        self.en_he_decoder = en_he_model.model.decoder
        self.lm_head = en_he_model.lm_head
        self.final_logits_bias = en_he_model.final_logits_bias

        # Freeze layers
        self._freeze_layers()

    def _freeze_layers(self):
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze the new linear layers
        for param in self.linear1.parameters():
            param.requires_grad = True
        for param in self.linear2.parameters():
            param.requires_grad = True

        # Unfreeze the custom decoder and encoder
        for param in self.custom_decoder1.parameters():
            param.requires_grad = True
        for param in self.custom_encoder2.parameters():
            param.requires_grad = True

    def forward(self, input_ids1, input_ids2, input_ids3, attention_mask1=None, attention_mask2=None, attention_mask3=None, llm=None, tokenizer2=None):
        # Ensure input tensors are of the correct data type
        input_ids1 = input_ids1.long()
        input_ids2 = input_ids2.long()
        input_ids3 = input_ids3.long()

        he_en_encoder_output = self.he_en_model_encoder(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        ).last_hidden_state

        inputs_embeds2 = self.custom_embedding(input_ids2, attention_mask2)

        x = self.custom_decoder1(
            inputs_embeds=inputs_embeds2,
            attention_mask=attention_mask2,
            encoder_hidden_states=he_en_encoder_output,
            encoder_attention_mask=attention_mask1,
        )[0]

        x = self.linear1(x)

        llm_attention_mask = create_opt_attention_mask(
            attention_mask=attention_mask2,
            input_shape=x.shape[:-1],
            inputs_embeds=x,
            past_key_values_length=0
        )

        # If llm is provided, process and print intermediate output
        if llm is not None and tokenizer2 is not None:
            hidden_states = llm.model.decoder.project_out(x)
            intermediate_logits = llm.lm_head(hidden_states)
            intermediate_tokens = torch.argmax(intermediate_logits, dim=-1)
            intermediate_text = tokenizer2.decode(intermediate_tokens[0], skip_special_tokens=True)
            print("Intermediate output (before main layers):", intermediate_text)

        for layer in self.main_layers:
            x = layer(hidden_states=x, attention_mask=llm_attention_mask)[0]

        # If llm is provided, process and print intermediate output again
        if llm is not None and tokenizer2 is not None:
            hidden_states = llm.model.decoder.project_out(x)
            intermediate_logits = llm.lm_head(hidden_states)
            intermediate_tokens = torch.argmax(intermediate_logits, dim=-1)
            intermediate_text = tokenizer2.decode(intermediate_tokens[0], skip_special_tokens=True)
            print("Intermediate output (after main layers):", intermediate_text)

        x = self.linear2(x)

        x = self.custom_encoder2(
            inputs_embeds=x,
            attention_mask=attention_mask2
        ).last_hidden_state

        # Teacher forcing
        x = self.en_he_decoder(
            encoder_hidden_states=x,
            encoder_attention_mask=attention_mask2,
            input_ids=input_ids3,
            attention_mask=attention_mask3,
        )[0]

        logits = self.lm_head(x) + self.final_logits_bias

        return logits

    def prepare_inputs(self, sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device):
        # Tokenizer 1: Hebrew sentence
        inputs_1 = tokenizer1(sentence, return_tensors="pt")
        input_ids_1 = inputs_1["input_ids"].to(device)
        attention_mask_1 = inputs_1["attention_mask"].to(device)

        # Translate the sentence
        with torch.no_grad():
            translated_ids = he_en_model.generate(input_ids=input_ids_1, attention_mask=attention_mask_1)
        translated_sentence = tokenizer1.decode(translated_ids[0], skip_special_tokens=True)

        # Tokenizer 2: English translation
        inputs_2 = tokenizer2(translated_sentence, return_tensors="pt")
        input_ids_2 = inputs_2["input_ids"].to(device)
        attention_mask_2 = inputs_2["attention_mask"].to(device)

        # Tokenizer 3: Full Hebrew sentence
        inputs_3 = tokenizer3(text_target=sentence, return_tensors="pt")
        input_ids_3 = inputs_3["input_ids"][:, :-1].to(device)
        attention_mask_3 = inputs_3["attention_mask"][:, :-1].to(device)

        # Add <pad> to the beginning
        batch_size, seq_length = input_ids_3.shape
        new_input_ids3 = torch.full((batch_size, seq_length + 1), tokenizer3.pad_token_id, dtype=input_ids_3.dtype,
                                    device=input_ids_3.device)
        new_input_ids3[:, 1:] = input_ids_3

        # Update attention mask
        new_attention_mask3 = torch.zeros((batch_size, seq_length + 1), dtype=attention_mask_3.dtype,
                                          device=attention_mask_3.device)
        new_attention_mask3[:, 1:] = attention_mask_3

        return {
            "input_ids_1": input_ids_1,
            "attention_mask_1": attention_mask_1,
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
            "input_ids_3": new_input_ids3,
            "attention_mask_3": new_attention_mask3
        }

    def generate(self, sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device, max_length=50,
                 temperature=1.0, top_k=50, top_p=0.95, llm=None):
        self.eval()

        # Prepare initial input tensors
        inputs = self.prepare_inputs(sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device)
        input_ids1 = inputs["input_ids_1"]
        input_ids2 = inputs["input_ids_2"]
        attention_mask1 = inputs["attention_mask_1"]
        attention_mask2 = inputs["attention_mask_2"]

        # Initialize the output sequence with the initial sentence from input_ids3
        generated_ids = inputs["input_ids_3"].clone()
        attention_mask3 = inputs["attention_mask_3"].clone()

        for _ in range(max_length):
            # Forward pass
            with torch.no_grad():
                outputs = self(
                    input_ids1=input_ids1,
                    input_ids2=input_ids2,
                    input_ids3=generated_ids,
                    attention_mask1=attention_mask1,
                    attention_mask2=attention_mask2,
                    attention_mask3=attention_mask3,
                    llm=llm,
                    tokenizer2=tokenizer2
                )

            # Get the next token logits
            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Check for NaN or inf values
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                print("Warning: NaN or inf values detected in next_token_logits")
                next_token_logits = torch.nan_to_num(next_token_logits, nan=0.0, posinf=1e6, neginf=-1e6)

            # Apply top-k filtering
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # Safety check
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < top_k_logits[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Create a boolean mask instead of direct indexing
                mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[mask] = float('-inf')

            # Sample the next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask3 = torch.cat([attention_mask3, torch.ones_like(next_token)], dim=1)

            # Convert the generated sequence to a sentence
            current_sentence = tokenizer3.decode(generated_ids[0], skip_special_tokens=True)

            # Use prepare_inputs to get updated inputs
            inputs = self.prepare_inputs(current_sentence, he_en_model, tokenizer1, tokenizer2, tokenizer3, device)
            input_ids1 = inputs["input_ids_1"]
            input_ids2 = inputs["input_ids_2"]
            attention_mask1 = inputs["attention_mask_1"]
            attention_mask2 = inputs["attention_mask_2"]

            # Check if we've generated an EOS token
            if next_token.item() == tokenizer3.eos_token_id:
                break

        return generated_ids

    @classmethod
    def load_pretrained(cls, checkpoint_path, he_en_model, en_he_model, llm_model, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            # This is a checkpoint saved by our Trainer
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # This might be a checkpoint saved by pytorch-lightning or other libraries
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's just the state dict
            state_dict = checkpoint

        model = cls(he_en_model, en_he_model, llm_model)

        # Load the state dict
        model.load_state_dict(state_dict)

        return model.to(device)
