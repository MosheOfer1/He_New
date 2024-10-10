import logging
import os
from datetime import datetime
import torch.nn as nn


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

    # Use a set to keep track of unique parameters
    unique_params = set()
    unique_trainable_params = set()

    def count_unique_params(module):
        for param in module.parameters():
            param_id = id(param)
            if param_id not in unique_params:
                unique_params.add(param_id)
                if param.requires_grad:
                    unique_trainable_params.add(param_id)

    # Recursively count unique parameters
    model.apply(count_unique_params)

    total_unique_params = sum(p.numel() for p in model.parameters() if id(p) in unique_params)
    total_unique_trainable_params = sum(p.numel() for p in model.parameters() if id(p) in unique_trainable_params)

    logger.info(f"\nTotal unique parameters: {total_unique_params:,}")
    logger.info(f"Unique trainable parameters: {total_unique_trainable_params:,}")
    logger.info(f"Percentage of unique trainable parameters: {total_unique_trainable_params / total_unique_params * 100:.2f}%")

    logger.info("\nDetailed Layer-wise Information:")
    for name, module in model.named_modules():
        param_count = sum(p.numel() for p in module.parameters())
        trainable_param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"\n{name}:")
        logger.info(f"  Total params: {param_count:,}")
        logger.info(f"  Trainable params: {trainable_param_count:,}")

        if isinstance(module, nn.TransformerEncoderLayer):
            mha = module.self_attn
            logger.info(f"  Attention Heads: {mha.num_heads if hasattr(mha, 'num_heads') else 'Not specified'}")
            logger.info(f"  Head Dimension: {mha.head_dim if hasattr(mha, 'head_dim') else 'Not specified'}")
        elif isinstance(module, nn.Linear):
            logger.info(f"  In features: {module.in_features}")
            logger.info(f"  Out features: {module.out_features}")

    # Optional: If you want to log the model summary to a file
    with open('detailed_model_summary.txt', 'w') as f:
        f.write(str(model))
    logger.info("Detailed model summary has been saved to 'detailed_model_summary.txt'")
    