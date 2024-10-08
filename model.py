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
        self.he_en_embeddings = he_en_model.model.shared
        self.he_en_encoder = he_en_model.model.encoder
        self.he_en_decoder = he_en_model.model.decoder

        # First custom transformer layer
        self.custom_layer1 = nn.Sequential(
            nn.Linear(he_en_model.config.hidden_size, llm_model.config.hidden_size),
            TransformerEncoderLayer(
                d_model=llm_model.config.hidden_size,
                nhead=llm_model.config.num_attention_heads,
                dim_feedforward=llm_model.config.ffn_dim,
                dropout=llm_model.config.hidden_dropout_prob,
                activation=llm_model.config.hidden_act,
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
                dropout=llm_model.config.hidden_dropout_prob,
                activation=llm_model.config.hidden_act,
                batch_first=True
            ),
            nn.Linear(llm_model.config.hidden_size, en_he_model.config.hidden_size)
        )

        # English-Hebrew components
        self.en_he_encoder = en_he_model.model.encoder
        self.en_he_decoder = en_he_model.model.decoder

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
        freeze_module(self.en_he_decoder)

        # Ensure custom layers are trainable
        for param in self.custom_layer1.parameters():
            param.requires_grad = True
        for param in self.custom_layer2.parameters():
            param.requires_grad = True

        # Ensure output projection is trainable
        for param in self.output_projection.parameters():
            param.requires_grad = True

    def forward(self, input_ids, en_target_ids, he_target_ids, he_attention_mask=None, en_attention_mask=None, llm_attention_mask=None):
        # Ensure input_ids is of type Long
        input_ids = input_ids.long()
        en_target_ids = en_target_ids.long()
        he_target_ids = he_target_ids.long()

        # 1. Hebrew-English encoding
        encoder_output = self.he_en_encoder(input_ids=input_ids, attention_mask=he_attention_mask).last_hidden_state

        # 2. Hebrew-English decoding with teacher forcing
        he_en_decoder_input_ids = torch.cat([
            torch.full((en_target_ids.shape[0], 1), self.he_en_decoder_start_token_id, device=en_target_ids.device),
            en_target_ids[:, :-1]
        ], dim=1)

        he_en_decoder_output = self.he_en_decoder(
            input_ids=he_en_decoder_input_ids,
            encoder_hidden_states=encoder_output,
            attention_mask=en_attention_mask
        ).last_hidden_state

        # 3. First custom layer
        x = self.custom_layer1(he_en_decoder_output)

        # 4. LLM processing
        for layer in self.main_layers:
            x = layer(hidden_states=x, attention_mask=llm_attention_mask)[0]

        # 5. Second custom layer
        x = self.custom_layer2(x)

        # 6. English-Hebrew encoding
        en_he_encoder_output = self.en_he_encoder(inputs_embeds=x, attention_mask=en_attention_mask).last_hidden_state

        # 7. English-Hebrew decoding with teacher forcing
        en_he_decoder_input_ids = torch.cat([
            torch.full((he_target_ids.shape[0], 1), self.en_he_decoder.config.decoder_start_token_id,
                       device=he_target_ids.device),
            he_target_ids[:, :-1]
        ], dim=1)

        final_output = self.en_he_decoder(
            input_ids=en_he_decoder_input_ids,
            encoder_hidden_states=en_he_encoder_output,
            attention_mask=he_attention_mask  # Use Hebrew attention mask for decoding
        ).last_hidden_state

        # 8. Final projection
        logits = self.output_projection(final_output)

        return logits
