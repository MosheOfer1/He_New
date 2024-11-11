import argparse
import torch
from torch import nn
from transformers import MarianMTModel, AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import logging
import os
import json
from model import CustomLLM
from data_set import create_dataloaders
from auto_encoder import DimensionAlignmentAutoencoder, AutoencoderPreTrainer


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


def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_texts(file_path):
    """Load texts from a file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, args):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        try:
            # Skip None batches (from collate_fn)
            if batch is None:
                continue

            # Move batch to device
            input_ids1 = batch['input_ids_1'].to(device)
            attention_mask1 = batch['attention_mask_1'].to(device)
            input_ids2 = batch['input_ids_2'].to(device)
            attention_mask2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)
            labels_attention_mask = batch['labels_attention_mask'].to(device)

            # Forward pass
            outputs = model(
                input_ids1=input_ids1,
                input_ids2=input_ids2,
                attention_mask1=attention_mask1,
                attention_mask2=attention_mask2,
                labels=labels,
                labels_attention_mask=labels_attention_mask
            )

            loss = outputs.loss

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\nWarning: NaN loss detected at batch {batch_idx}")
                continue

            total_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Log to wandb
            if args.use_wandb and batch_idx % args.log_interval == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': batch_idx
                })
        except Exception as e:
            print(e)
            continue


    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            try:
                # Skip None batches
                if batch is None:
                    continue

                # Sequence length monitoring
                seq_lengths = {
                    'input_ids_1': batch['input_ids_1'].size(),
                    'input_ids_2': batch['input_ids_2'].size(),
                    'labels': batch['labels'].size(),
                }

                # Log if sequences are unusually long
                max_expected_length = 512
                for key, size in seq_lengths.items():
                    if size[1] > max_expected_length:
                        print(
                            f"\nWarning: {key} sequence length {size[1]} exceeds {max_expected_length} at batch {batch_idx}")
                        continue

                input_ids1 = batch['input_ids_1'].to(device)
                attention_mask1 = batch['attention_mask_1'].to(device)
                input_ids2 = batch['input_ids_2'].to(device)
                attention_mask2 = batch['attention_mask_2'].to(device)
                labels = batch['labels'].to(device)
                labels_attention_mask = batch['labels_attention_mask'].to(device)

                outputs = model(
                    input_ids1=input_ids1,
                    input_ids2=input_ids2,
                    attention_mask1=attention_mask1,
                    attention_mask2=attention_mask2,
                    labels=labels,
                    labels_attention_mask=labels_attention_mask
                )

                total_loss += outputs.loss.item()
            except Exception as e:
                print(e)
                continue

    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, save_path, args):
    """Save model checkpoint with all necessary information"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'args': vars(args)
    }

    torch.save(checkpoint, save_path)


def setup_autoencoder(args, he_en_model, llm_model, tokenizer1, device, train_texts):
    """
    Setup autoencoder: either load pre-trained or train new one
    """
    # Create the autoencoder
    autoencoder = DimensionAlignmentAutoencoder(
        input_dim=he_en_model.config.d_model,
        target_dim=llm_model.config.hidden_size
    ).to(device)  # Move to device immediately after creation

    autoencoder_path = os.path.join(args.output_dir, 'autoencoder', 'best_autoencoder.pt')

    # If load_pretrained_autoencoder flag is set and the file exists
    if args.load_pretrained_autoencoder and os.path.exists(args.autoencoder_path):
        print(f"Loading pre-trained autoencoder from {args.autoencoder_path}")
        checkpoint = torch.load(args.autoencoder_path, map_location=device, weights_only=True)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder = autoencoder.to(device)  # Ensure it's on the right device after loading

    # If train_autoencoder flag is set
    elif args.train_autoencoder:
        print("Training new autoencoder...")
        trainer = AutoencoderPreTrainer(
            autoencoder=autoencoder.encoder,
            he_en_model=he_en_model,
            tokenizer1=tokenizer1,
            device=device
        )

        best_loss = trainer.train(
            sentences=train_texts,
            num_epochs=args.autoencoder_epochs,
            batch_size=args.autoencoder_batch_size,
            learning_rate=args.autoencoder_lr,
            save_dir=os.path.join(args.output_dir, 'autoencoder')
        )
        print(f"Autoencoder training completed with best loss: {best_loss}")

        # Load the best model weights after training
        checkpoint = torch.load(autoencoder_path, map_location=device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder = autoencoder.to(device)  # Ensure it's on the right device after loading

    else:
        raise ValueError("Either load_pretrained_autoencoder must be True with a valid path, "
                         "or train_autoencoder must be True")

    # Final verification of device placement
    print(f"Verifying autoencoder device: {next(autoencoder.parameters()).device}")

    return autoencoder


def main(args):
    # Setup logging
    logger = setup_logging(args.output_dir)
    logger.info(f"Starting training with args: {args}")

    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            config=args
        )

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizers and models
    logger.info("Loading tokenizers and models...")
    tokenizer1 = AutoTokenizer.from_pretrained(args.he_en_model_name)
    tokenizer2 = AutoTokenizer.from_pretrained(args.llm_model_name)

    he_en_model = MarianMTModel.from_pretrained(args.he_en_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model_name)

    # Move models to device
    he_en_model = he_en_model.to(device)
    llm_model = llm_model.to(device)

    # Load training texts
    logger.info("Loading training texts...")
    train_texts = load_texts(args.train_hebrew_texts)
    logger.info(f"Loaded {len(train_texts)} training texts")


    # Setup autoencoder
    autoencoder = setup_autoencoder(
        args=args,
        he_en_model=he_en_model,
        llm_model=llm_model,
        tokenizer1=tokenizer1,
        device=device,
        train_texts=train_texts
    )

    # Initialize custom model with the prepared autoencoder
    model = CustomLLM(
        he_en_model=he_en_model,
        llm_model=llm_model,
        align_he_en=autoencoder,
        freeze_he_en=args.freeze_he_en,
        freeze_llm=args.freeze_llm,
        freeze_decoder=args.freeze_decoder,
        freeze_alignment=args.freeze_alignment,
        pad_token_id=tokenizer2.pad_token_id
    ).to(device)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        sentences=train_texts,
        he_en_model=he_en_model,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        batch_size=args.batch_size,
        train_split=args.train_split,
        device=device
    )

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs * len(train_loader)
    )

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')

    # Track training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'best_epoch': 0
    }
    print_model_info(model, logger)
    for epoch in range(args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, args)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Log metrics
        logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                save_path=os.path.join(args.output_dir, 'best_model.pt'),
                args=args
            )
            logger.info(f"Saved best model at epoch {epoch}")

        # Save latest model
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            save_path=os.path.join(args.output_dir, 'latest_model.pt'),
            args=args
        )

    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    logger.info(f"Saved training history to {history_path}")

    logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f} at epoch {history['best_epoch']}")
    if args.use_wandb:
        wandb.finish()


# Add these to your argument parser
def add_autoencoder_args(parser):
    group = parser.add_argument_group('Autoencoder')

    # Control flags
    group.add_argument('--train_autoencoder', action='store_true',
                       help='Whether to train a new autoencoder')
    group.add_argument('--load_pretrained_autoencoder', action='store_true',
                       help='Whether to load a pre-trained autoencoder')
    group.add_argument('--autoencoder_path', type=str, default=None,
                       help='Path to pre-trained autoencoder checkpoint')

    # Training parameters
    group.add_argument('--autoencoder_epochs', type=int, default=100,
                       help='Number of epochs for autoencoder training')
    group.add_argument('--autoencoder_batch_size', type=int, default=32,
                       help='Batch size for autoencoder training')
    group.add_argument('--autoencoder_lr', type=float, default=1e-4,
                       help='Learning rate for autoencoder training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Custom LLM')

    # Model parameters
    parser.add_argument('--he_en_model_name', type=str, default="Helsinki-NLP/opus-mt-tc-big-he-en",
                        help='Hebrew-English model name or path')
    parser.add_argument('--llm_model_name', type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help='LLM model name or path')
    parser.add_argument('--freeze_he_en', action='store_true',
                        help='Freeze Hebrew-English encoder')
    parser.add_argument('--freeze_llm', action='store_true',
                        help='Freeze main LLM model')
    parser.add_argument('--freeze_decoder', action='store_true',
                        help='Freeze custom decoder')
    parser.add_argument('--freeze_alignment', action='store_true',
                        help='Freeze alignment layer')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Training data split ratio')

    # Data parameters
    parser.add_argument('--train_hebrew_texts', type=str, required=True,
                        help='Path to Hebrew training texts')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')

    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                        help='Output directory for checkpoints')

    # Logging parameters
    parser.add_argument('--use_wandb', action='store_true',
                        help='Whether to use Weights & Biases logging')
    parser.add_argument('--wandb_project_name', type=str, default='custom_llm',
                        help='W&B project name')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval')
    add_autoencoder_args(parser)  # Add autoencoder-specific arguments

    args = parser.parse_args()
    main(args)
