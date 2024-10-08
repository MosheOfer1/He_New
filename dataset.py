import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, stride=256, eval_split=0.1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()

        self.tokenized_data = self.tokenizer.encode(self.data)

        # Create sliding windows
        self.windows = [self.tokenized_data[i:i + max_length] for i in
                        range(0, len(self.tokenized_data) - max_length + 1, stride)]

        # Split data into train and eval
        split_point = int(len(self.windows) * (1 - eval_split))
        self.train_windows = self.windows[:split_point]
        self.eval_windows = self.windows[split_point:]

    def __len__(self):
        return len(self.train_windows)

    def __getitem__(self, idx):
        window = self.train_windows[idx]
        x = torch.tensor(window[:-1], dtype=torch.long)
        y = torch.tensor(window[1:], dtype=torch.long)
        return x, y

    def get_eval_data(self):
        eval_dataset = []
        for window in self.eval_windows:
            x = torch.tensor(window[:-1], dtype=torch.long)
            y = torch.tensor(window[1:], dtype=torch.long)
            eval_dataset.append((x, y))
        return eval_dataset


def create_dataloaders(dataset, batch_size):
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eval_dataset = dataset.get_eval_data()
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, eval_dataloader
