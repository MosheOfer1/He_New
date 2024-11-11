import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MarianMTModel, PreTrainedTokenizer
import random


class TextDataset(Dataset):
    def __init__(self, sentences, he_en_model: MarianMTModel, tokenizer1: PreTrainedTokenizer,
                 tokenizer2: PreTrainedTokenizer, device: str, max_length=512):
        self.sentences = sentences
        self.he_en_model = he_en_model
        self.tokenizer1 = tokenizer1  # Hebrew-English model tokenizer
        self.tokenizer2 = tokenizer2  # LLM tokenizer
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        original_sentence = self.sentences[idx]

        # Tokenizer 1: Hebrew sentence with first tokenizer (for translation model)
        inputs_1 = self.tokenizer1(original_sentence, return_tensors="pt", max_length=self.max_length,
            truncation=True)
        input_ids_1 = inputs_1["input_ids"].to(self.device)
        attention_mask_1 = inputs_1["attention_mask"].to(self.device)

        # Also tokenize Hebrew sentence with second tokenizer (for target)
        target_tokens = self.tokenizer2(original_sentence, return_tensors="pt")
        target_ids = target_tokens["input_ids"].to(self.device)
        target_attention_mask = target_tokens["attention_mask"].to(self.device)

        # Remove the second-to-last token but keep the last token (e.g., </s>)
        if input_ids_1.size(1) > 2:
            input_ids_1 = torch.cat((input_ids_1[:, :-2], input_ids_1[:, -1:]), dim=1).to(self.device)
            attention_mask_1 = torch.cat((attention_mask_1[:, :-2], attention_mask_1[:, -1:]), dim=1).to(self.device)

        # Translate the modified sentence on the fly
        translated_sentence = self.he_en_model.generate(input_ids=input_ids_1, attention_mask=attention_mask_1)
        translated_sentence = self.tokenizer1.decode(translated_sentence[0], skip_special_tokens=True)

        # Tokenizer 2: English translation
        inputs_2 = self.tokenizer2(translated_sentence, return_tensors="pt")
        input_ids_2 = inputs_2["input_ids"].to(self.device)
        attention_mask_2 = inputs_2["attention_mask"].to(self.device)

        return {
            "input_ids_1": input_ids_1.squeeze(),
            "attention_mask_1": attention_mask_1.squeeze(),
            "input_ids_2": input_ids_2.squeeze(),
            "attention_mask_2": attention_mask_2.squeeze(),
            "labels": target_ids.squeeze(),  # Hebrew text tokenized with tokenizer2
            "labels_attention_mask": target_attention_mask.squeeze()
        }

    def collate_fn(self, batch):
        def ensure_1d(tensor):
            if tensor.dim() == 0:
                return tensor.unsqueeze(0)
            return tensor

        input_ids_1 = [ensure_1d(item["input_ids_1"]) for item in batch if item["input_ids_1"].numel() > 0]
        attention_mask_1 = [ensure_1d(item["attention_mask_1"]) for item in batch if item["attention_mask_1"].numel() > 0]
        input_ids_2 = [ensure_1d(item["input_ids_2"]) for item in batch if item["input_ids_2"].numel() > 0]
        attention_mask_2 = [ensure_1d(item["attention_mask_2"]) for item in batch if item["attention_mask_2"].numel() > 0]
        labels = [ensure_1d(item["labels"]) for item in batch if item["labels"].numel() > 0]
        labels_attention_mask = [ensure_1d(item["labels_attention_mask"]) for item in batch if item["labels_attention_mask"].numel() > 0]

        # Check if any of the lists are empty after filtering
        if not all([input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels, labels_attention_mask]):
            return None

        # Pad each batch according to its respective tokenizer
        try:
            padded_1 = torch.nn.utils.rnn.pad_sequence(input_ids_1, batch_first=True, padding_value=self.tokenizer1.pad_token_id)
            padded_attention_1 = torch.nn.utils.rnn.pad_sequence(attention_mask_1, batch_first=True, padding_value=0)
            padded_2 = torch.nn.utils.rnn.pad_sequence(input_ids_2, batch_first=True, padding_value=self.tokenizer2.pad_token_id)
            padded_attention_2 = torch.nn.utils.rnn.pad_sequence(attention_mask_2, batch_first=True, padding_value=0)
            padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer2.pad_token_id)
            padded_labels_attention_mask = torch.nn.utils.rnn.pad_sequence(labels_attention_mask, batch_first=True, padding_value=0)
        except Exception as e:
            return None

        return {
            "input_ids_1": padded_1,
            "attention_mask_1": padded_attention_1,
            "input_ids_2": padded_2,
            "attention_mask_2": padded_attention_2,
            "labels": padded_labels,
            "labels_attention_mask": padded_labels_attention_mask
        }


def create_dataloaders(sentences, he_en_model, tokenizer1, tokenizer2, batch_size, train_split, device):
    # Split data into training and evaluation sets
    random.shuffle(sentences)
    split_idx = int(train_split * len(sentences))
    train_sentences = sentences[:split_idx]
    eval_sentences = sentences[split_idx:]

    # Create Datasets
    train_dataset = TextDataset(train_sentences, he_en_model, tokenizer1, tokenizer2, device)
    eval_dataset = TextDataset(eval_sentences, he_en_model, tokenizer1, tokenizer2, device)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn)

    return train_loader, eval_loader