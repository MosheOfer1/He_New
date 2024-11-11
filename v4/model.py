import os

import torch
import torch.nn as nn
from transformers import MarianConfig, PreTrainedModel
from transformers.models.marian.modeling_marian import MarianDecoder
from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from auto_encoder import DimensionAlignmentAutoencoder


@dataclass
class CustomLLMOutput:
    """
    Output type for CustomLLM model.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    intermediate_text: Optional[str] = None


class CustomLLM(nn.Module):
    def __init__(
            self,
            he_en_model: PreTrainedModel,
            llm_model: PreTrainedModel,
            align_he_en: DimensionAlignmentAutoencoder=None,
            freeze_he_en: bool = True,
            freeze_llm: bool = True,
            freeze_decoder: bool = False,
            freeze_alignment: bool = True,
            pad_token_id=-100
    ):
        """
        Initialize the Custom LLM model.

        Args:
            he_en_model: Pretrained Hebrew-English translation model
            llm_model: Pretrained language model
            freeze_he_en: Whether to freeze the Hebrew-English encoder
            freeze_llm: Whether to freeze the main LLM model
            freeze_decoder: Whether to freeze the custom decoder
            freeze_alignment: Whether to freeze the alignment layer
        """
        super().__init__()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Save config values
        self.he_en_config = he_en_model.config
        self.llm_config = llm_model.config
        self.pad_token_id = pad_token_id

        # Hebrew-English components
        self.he_en_model_encoder = he_en_model.model.encoder

        # Create custom decoder config
        decoder_config = self._create_decoder_config()

        # Initialize custom decoder
        self.custom_decoder1 = self._initialize_decoder(decoder_config, llm_model.model.embed_tokens)

        # Dimension alignment layer
        if align_he_en is None:
            self.align_he_en = DimensionAlignmentAutoencoder(
                input_dim=he_en_model.config.d_model,
                target_dim=llm_model.config.hidden_size
            )
        else:
            self.align_he_en = align_he_en

            # Main LLM model
        self.main_model = llm_model.model

        # Initialize second custom decoder2
        self.custom_decoder2 = self._initialize_decoder(decoder_config, llm_model.model.embed_tokens)

        # The head
        self.lm_head = llm_model.lm_head

        # Freeze layers based on parameters
        self._freeze_layers(
            freeze_he_en=freeze_he_en,
            freeze_llm=freeze_llm,
            freeze_decoder=freeze_decoder,
            freeze_alignment=freeze_alignment
        )

        # Initialize weights where needed
        self._initialize_weights()
        self.main_model.set_input_embeddings(None)

    def _create_decoder_config(self) -> MarianConfig:
        """Create config for the custom decoder."""
        return MarianConfig(
            d_model=self.llm_config.hidden_size,
            encoder_attention_heads=self.he_en_config.encoder_attention_heads,
            encoder_ffn_dim=self.he_en_config.encoder_ffn_dim,
            encoder_layers=self.he_en_config.encoder_layers,
            decoder_attention_heads=self.he_en_config.decoder_attention_heads,
            decoder_ffn_dim=self.he_en_config.decoder_ffn_dim,
            decoder_layers=self.he_en_config.decoder_layers,
            max_length=self.he_en_config.max_length,
            vocab_size=self.he_en_config.vocab_size,
            scale_embedding=self.he_en_config.scale_embedding,
            pad_token_id=self.he_en_config.pad_token_id,
            eos_token_id=self.he_en_config.eos_token_id,
            decoder_start_token_id=self.he_en_config.decoder_start_token_id,
        )

    def _initialize_decoder(self, config: MarianConfig, embed_tokens) -> MarianDecoder:
        """Initialize the custom decoder."""
        decoder = MarianDecoder(config)
        decoder.set_input_embeddings(embed_tokens)  # Clear default embeddings
        return decoder

    def _freeze_layers(
            self,
            freeze_he_en: bool,
            freeze_llm: bool,
            freeze_decoder: bool,
            freeze_alignment: bool
    ):
        """
        Freeze specified components of the model.
        """

        def freeze_params(module: nn.Module):
            for param in module.parameters():
                param.requires_grad = False

        if freeze_he_en:
            self.logger.info("Freezing Hebrew-English encoder")
            freeze_params(self.he_en_model_encoder)

        if freeze_llm:
            self.logger.info("Freezing main LLM model")
            freeze_params(self.main_model)
            # Unfreeze final layer for fine-tuning if needed
            for param in self.main_model.layers[-1].parameters():
                param.requires_grad = True

        if freeze_decoder:
            self.logger.info("Freezing custom decoder")
            freeze_params(self.custom_decoder1)

        if freeze_alignment:
            self.logger.info("Freezing alignment autoencoder")
            freeze_params(self.align_he_en.encoder)
            freeze_params(self.align_he_en.decoder)

    def _initialize_weights(self):
        """Initialize weights for unfrozen layers."""

        # Initialize unfrozen decoder layers
        for param in self.custom_decoder1.parameters():
            if param.requires_grad:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)

    def forward(
            self,
            input_ids1: torch.Tensor,
            input_ids2: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            attention_mask1: Optional[torch.Tensor] = None,
            attention_mask2: Optional[torch.Tensor] = None,
            labels_attention_mask: Optional[torch.Tensor] = None,
            llm: Optional[PreTrainedModel] = None,
            tokenizer2: Optional[Any] = None,
            output_hidden_states: bool = False,
            output_attentions: bool = False,
            return_dict: bool = True,
    ) -> Union[CustomLLMOutput, Tuple]:
        """
        Forward pass of the model.

        Args:
            input_ids1: Hebrew input ids
            input_ids2: English input ids
            labels: Target labels
            labels_attention_mask: Attention mask for target labels
            attention_mask1: Attention mask for Hebrew input
            attention_mask2: Attention mask for English input
            llm: Optional LLM model for intermediate output
            tokenizer2: Optional tokenizer for intermediate output
            output_hidden_states: Whether to return hidden states
            output_attentions: Whether to return attention weights
            return_dict: Whether to return a CustomLLMOutput object

        Returns:
            Model outputs either as a CustomLLMOutput object or tuple
        """
        # Ensure input tensors are of the correct data type
        input_ids1 = input_ids1.long()
        input_ids2 = input_ids2.long()
        if labels is not None:
            labels = labels.long()

        # Process through Hebrew-English encoder
        he_en_outputs = self.he_en_model_encoder(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        he_en_encoder_output = he_en_outputs.last_hidden_state

        # Align Hebrew-English encoder output dimensions with LLM
        he_en_encoder_output, _ = self.align_he_en(he_en_encoder_output, True)

        # Process through custom decoder
        decoder_outputs = self.custom_decoder1(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            encoder_hidden_states=he_en_encoder_output,
            encoder_attention_mask=attention_mask1,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        x = decoder_outputs[0]

        # Get intermediate output if requested
        intermediate_text = None
        if llm is not None and tokenizer2 is not None:
            hidden_states = llm.model.decoder.project_out(x)
            intermediate_logits = llm.lm_head(hidden_states)
            intermediate_tokens = torch.argmax(intermediate_logits, dim=-1)
            intermediate_text = tokenizer2.decode(intermediate_tokens[0], skip_special_tokens=True)
            self.logger.info(f"Intermediate output: {intermediate_text}")

        # Process through main LLM
        llm_outputs = self.main_model(
            inputs_embeds=x,
            attention_mask=attention_mask2,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        ).last_hidden_state

        # Process through custom decoder
        decoder_outputs = self.custom_decoder2(
            input_ids=labels,
            attention_mask=labels_attention_mask,
            encoder_hidden_states=llm_outputs,
            encoder_attention_mask=attention_mask2,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        hidden_states = decoder_outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift the logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross entropy loss
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_token_id
            )
        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        return CustomLLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            intermediate_text=intermediate_text
        )


    def generate(
            self,
            input_ids1: torch.Tensor,
            input_ids2: torch.Tensor,
            attention_mask1: Optional[torch.Tensor] = None,
            attention_mask2: Optional[torch.Tensor] = None,
            **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.

        Args:
            input_ids1: Hebrew input ids
            input_ids2: English input ids
            attention_mask1: Attention mask for Hebrew input
            attention_mask2: Attention mask for English input
            **generate_kwargs: Additional arguments for generation

        Returns:
            Generated token ids
        """
        # Process through Hebrew-English encoder
        he_en_encoder_output = self.he_en_model_encoder(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
        ).last_hidden_state

        # Align dimensions
        he_en_encoder_output = self.align_he_en(he_en_encoder_output)

        # Get embeddings for the English input
        inputs_embeds2 = self.custom_embedding(input_ids2)

        # Process through custom decoder
        x = self.custom_decoder1(
            inputs_embeds=inputs_embeds2,
            attention_mask=attention_mask2,
            encoder_hidden_states=he_en_encoder_output,
            encoder_attention_mask=attention_mask1,
        )[0]

        # Generate using main model
        return self.main_model.generate(
            inputs_embeds=x,
            attention_mask=attention_mask2,
            **generate_kwargs
        )

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        os.makedirs(save_directory, exist_ok=True)

        # Save configs
        self.he_en_config.save_pretrained(save_directory)
        self.llm_config.save_pretrained(save_directory)

        # Save model state
        model_state = {
            'state_dict': self.state_dict(),
            'he_en_config': self.he_en_config,
            'llm_config': self.llm_config
        }
        torch.save(model_state, os.path.join(save_directory, 'model_state.pt'))

    @classmethod
    def from_pretrained(
            cls,
            he_en_model: PreTrainedModel,
            llm_model: PreTrainedModel,
            model_path: str
    ) -> 'CustomLLM':
        """Load model from directory."""
        model_state = torch.load(os.path.join(model_path, 'model_state.pt'))
        model = cls(he_en_model, llm_model)
        model.load_state_dict(model_state['state_dict'])
        return model