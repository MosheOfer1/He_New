import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import traceback


def log_error(logger, error, batch):
    logger.error(f"Error during forward pass: {str(error)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            logger.error(f"{key} shape: {value.shape}, device: {value.device}")
            logger.error(f"{key} min: {value.min().item()}, max: {value.max().item()}")
    logger.error("Skipping this batch due to error.")


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
    logger.info("\nDetailed Model Architecture:")
    logger.info(str(model))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")

    logger.info("\nDetailed Layer-wise Information:")
    for name, module in model.named_children():
        param_count = sum(p.numel() for p in module.parameters())
        trainable_param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"\n{name}:")
        logger.info(f"  Total params: {param_count:,}")
        logger.info(f"  Trainable params: {trainable_param_count:,}")

        if isinstance(module, nn.Sequential):
            for idx, layer in enumerate(module):
                if isinstance(layer, nn.TransformerEncoderLayer):
                    mha = layer.self_attn
                    logger.info(f"    Attention Heads: {mha.num_heads}")
                    logger.info(f"    Head Dimension: {mha.head_dim}")
                elif isinstance(layer, nn.Linear):
                    logger.info(f"    In features: {layer.in_features}")
                    logger.info(f"    Out features: {layer.out_features}")

        elif isinstance(module, nn.ModuleList):
            logger.info(f"  Number of layers: {len(module)}")
            if len(module) > 0:
                if hasattr(module[0], 'self_attn'):
                    mha = module[0].self_attn
                    logger.info(
                        f"    Attention Heads: {mha.num_heads if hasattr(mha, 'num_heads') else 'Not specified'}")
                    logger.info(f"    Head Dimension: {mha.head_dim if hasattr(mha, 'head_dim') else 'Not specified'}")

    # Optional: If you want to log the model summary to a file
    with open('logs/detailed_model_summary.txt', 'w') as f:
        f.write(str(model))
    logger.info("Detailed model summary has been saved to 'detailed_model_summary.txt'")


def create_opt_attention_mask(
        attention_mask,
        input_shape,
        inputs_embeds,
        past_key_values_length: int = 0,
        sliding_window=None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask
