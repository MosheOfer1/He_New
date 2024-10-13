import torch
import torch.nn as nn
from transformers.models.marian.modeling_marian import MarianDecoder, MarianEncoder
from transformers.models.opt.modeling_opt import OPTAttention

from v2.utils import create_opt_attention_mask


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
        self.custom_encoder2.embed_positions = llm_model.model.decoder.embed_positions

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

    def forward(self, input_ids1, input_ids2, input_ids3, attention_mask1=None, attention_mask2=None, attention_mask3=None):
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

        for layer in self.main_layers:
            x = layer(hidden_states=x, attention_mask=llm_attention_mask)[0]

        x = self.linear2(x)

        x = self.custom_encoder2(
            inputs_embeds=x,
            attention_mask=attention_mask2
        ).last_hidden_state

        x = self.en_he_decoder(
            inputs_embeds=x,
            attention_mask=attention_mask2,
            decoder_input_ids=input_ids3,
            decoder_attention_mask=attention_mask3,
        )[0]

        logits = self.lm_head(x) + self.final_logits_bias

        return logits

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
