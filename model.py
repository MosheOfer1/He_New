import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer


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
        self.he_en_model = he_en_model.model

        # First custom transformer layer
        self.custom_layer1 = nn.Sequential(
            nn.Linear(he_en_model.config.hidden_size, llm_model.config.hidden_size),
            TransformerEncoderLayer(
                d_model=llm_model.config.hidden_size,
                nhead=llm_model.config.num_attention_heads,
                dim_feedforward=llm_model.config.ffn_dim,
                dropout=llm_model.config.dropout,
                activation="relu",
                batch_first=True
            )
        )

        # LLM layers (main body of the model)
        self.main_layers = llm_model.model.decoder.layers

        # Second custom transformer layer
        self.custom_layer2 = nn.Sequential(
            TransformerEncoderLayer(
                d_model=llm_model.config.hidden_size,
                nhead=llm_model.config.num_attention_heads,
                dim_feedforward=llm_model.config.ffn_dim,
                dropout=llm_model.config.dropout,
                activation="relu",
                batch_first=True
            ),
            nn.Linear(llm_model.config.hidden_size, en_he_model.config.hidden_size)
        )

        # English-Hebrew components
        self.en_he_model = en_he_model.model

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

        # Create a new instance of CustomLLM
        vocab_size = state_dict['output_projection.weight'].size(0)
        bottleneck_size = state_dict['factorized_embedding.0.weight'].size(1)

        model = cls(he_en_model, en_he_model, llm_model, vocab_size, bottleneck_size)

        # Load the state dict
        model.load_state_dict(state_dict)

        return model.to(device)

    def _freeze_layers(self):
        # Helper function to freeze all parameters in a module
        def freeze_module(module):
            for param in module.parameters():
                param.requires_grad = False

        # Freeze Hebrew-English components
        freeze_module(self.he_en_model)

        # Freeze LLM layers
        for layer in self.main_layers:
            freeze_module(layer)

        # Freeze English-Hebrew components
        freeze_module(self.en_he_model)

        # Ensure custom layers are trainable
        for param in self.custom_layer1.parameters():
            param.requires_grad = True
        for param in self.custom_layer2.parameters():
            param.requires_grad = True

        # Ensure output projection is trainable
        for param in self.output_projection.parameters():
            param.requires_grad = True

    def forward(self, input_ids, en_target_ids, he_attention_mask=None, en_attention_mask=None,
                llm_attention_mask=None):
        # Ensure input tensors are of the correct data type
        input_ids = input_ids.long()
        en_target_ids = en_target_ids.long()

        # Phase 1: Hebrew to English Translation
        # Process the input through the Hebrew-English model
        he_en_decoder_output = self.he_en_model(
            input_ids=input_ids,
            attention_mask=he_attention_mask,
            decoder_input_ids=en_target_ids,
            decoder_attention_mask=en_attention_mask,
        ).last_hidden_state

        # Phase 2: Custom Processing
        # Apply the first custom layer to refine the translation output
        x = self.custom_layer1(he_en_decoder_output)

        # Phase 3: Language Model Enhancement
        # Pass the refined output through the main language model layers
        for layer in self.main_layers:
            x = layer(hidden_states=x, attention_mask=llm_attention_mask)[0]

        # Phase 4: Custom Processing
        # Prepare the enhanced representation for the next phase
        inputs_embeds = self.custom_layer2(x)

        # Phase 5: English to Hebrew Translation
        # Prepare the decoder input for the English-Hebrew model
        en_he_decoder_input_ids = torch.cat([
            torch.full((input_ids.shape[0], 1), self.en_he_model.decoder.config.decoder_start_token_id,
                       device=input_ids.device),
            input_ids[:, :-1]
        ], dim=1)

        # Process the enhanced representation through the English-Hebrew model
        final_output = self.en_he_model(
            inputs_embeds=inputs_embeds,
            attention_mask=en_attention_mask,
            decoder_input_ids=en_he_decoder_input_ids,
            decoder_attention_mask=he_attention_mask,
        ).last_hidden_state

        # Phase 5: Final Output Generation
        # Project the final hidden states to the output vocabulary space
        logits = self.output_projection(final_output)

        return logits
