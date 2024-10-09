import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, eval_split=0.1, max_length=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

        with open(file_path, 'r', encoding='utf-8') as f:
            self.sentences = [line.strip() for line in f if line.strip()]

        # Split data into train and eval
        split_point = int(len(self.sentences) * (1 - eval_split))
        self.train_sentences = self.sentences[:split_point]
        self.eval_sentences = self.sentences[split_point:]

    def __len__(self):
        return len(self.train_sentences)

    def __getitem__(self, idx):
        sentence = self.train_sentences[idx]
        tokenized = self.tokenizer.encode(sentence, max_length=self.max_length, truncation=True)
        input_ids = torch.tensor(tokenized[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokenized[1:], dtype=torch.long)
        return input_ids, target_ids

    def get_eval_data(self):
        eval_data = []
        for sentence in self.eval_sentences:
            tokenized = self.tokenizer.encode(sentence, max_length=self.max_length, truncation=True)
            input_ids = torch.tensor(tokenized[:-1], dtype=torch.long)
            target_ids = torch.tensor(tokenized[1:], dtype=torch.long)
            eval_data.append((input_ids, target_ids))
        return eval_data

    def collate_batch(self, batch):
        # Separate inputs and targets
        input_ids, target_ids = zip(*batch)

        # Pad sequences
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_target_ids = pad_sequence(target_ids, batch_first=True, padding_value=self.pad_token_id)

        # Create attention masks
        attention_mask = torch.zeros_like(padded_input_ids).masked_fill(padded_input_ids != 0, 1)

        return padded_input_ids, padded_target_ids, attention_mask


def create_dataloaders(dataset, batch_size):
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate_batch
    )

    eval_dataset = dataset.get_eval_data()
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_batch
    )

    return train_dataloader, eval_dataloader
