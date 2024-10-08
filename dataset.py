import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, eval_split=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()

        self.tokenized_data = tokenizer.encode(self.data)

        # Split data into train and eval
        split_point = int(len(self.tokenized_data) * (1 - eval_split))
        self.train_data = self.tokenized_data[:split_point]
        self.eval_data = self.tokenized_data[split_point:]

    def __len__(self):
        return len(self.train_data) - self.max_length

    def __getitem__(self, idx):
        chunk = self.train_data[idx:idx + self.max_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def get_eval_data(self):
        eval_dataset = []
        for i in range(0, len(self.eval_data) - self.max_length, self.max_length):
            chunk = self.eval_data[i:i + self.max_length + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)
            eval_dataset.append((x, y))
        return eval_dataset


def create_dataloaders(dataset, batch_size):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = dataset.get_eval_data()
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, eval_dataloader
