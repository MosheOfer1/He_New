import traceback

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
import copy

from utils import print_progress_bar


class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.history = {"lr": [], "loss": []}
        self.best_loss = None

    def range_test(self, train_loader, end_lr=10, num_iter=100, step_mode="exp", smooth_f=0.05, diverge_th=5):
        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter) if step_mode == "exp" else LinearLR(
            self.optimizer, end_lr, num_iter)

        iterator = iter(train_loader)
        for iteration in range(num_iter):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            try:
                loss = self._train_batch(batch)

                # Compute the smoothed loss
                avg_loss = self.smooth(loss.item())
                self.history["lr"].append(lr_scheduler.get_lr()[0])
                self.history["loss"].append(avg_loss)

                if self.best_loss is None:
                    self.best_loss = avg_loss
                elif avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                elif avg_loss > self.diverge_th * self.best_loss:
                    print("\nStopping early, the loss has diverged")
                    break

            except Exception as e:
                print(f"\nError encountered in iteration {iteration}: {str(e)}")
                print(traceback.format_exc())
                print("Skipping this batch and continuing...")
                continue

            lr_scheduler.step()

            # Update progress bar
            print_progress_bar(iteration + 1, num_iter, 1, 1, prefix='LR Finder:',
                               suffix=f'Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_lr()[0]:.6f}')

        print()  # Print a newline after the progress bar is complete
        return self.history

    def _train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._forward_pass(batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def _forward_pass(self, batch):
        try:
            input_ids_1 = batch['input_ids_1'].to(self.device)
            input_ids_2 = batch['input_ids_2'].to(self.device)
            input_ids_3 = batch['input_ids_3'].to(self.device)
            attention_mask_1 = batch['attention_mask_1'].to(self.device)
            attention_mask_2 = batch['attention_mask_2'].to(self.device)
            attention_mask_3 = batch['attention_mask_3'].to(self.device)

            outputs = self.model(input_ids_1, input_ids_2, input_ids_3,
                                 attention_mask1=attention_mask_1,
                                 attention_mask2=attention_mask_2,
                                 attention_mask3=attention_mask_3)

            # Assuming the loss is calculated using input_ids_3 as targets
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), input_ids_3.view(-1))
            return loss
        except Exception as e:
            print(f"\nError in _forward_pass: {str(e)}")
            print(f"Batch keys: {batch.keys()}")
            print(f"Batch shapes: {[batch[k].shape for k in batch.keys()]}")
            raise

    def smooth(self, loss):
        if self.best_loss is None:
            return loss
        return self.best_loss * self.smooth_f + loss * (1 - self.smooth_f)

    def plot(self, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.savefig(save_path)
        plt.close()


class LinearLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


def find_best_lr(model, train_dataloader, device, save_path):
    model_copy = copy.deepcopy(model)
    optimizer = Adam(model_copy.parameters(), lr=1e-7, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model_copy.en_he_decoder.config.pad_token_id)
    lr_finder = LRFinder(model_copy, optimizer, criterion, device)

    history = lr_finder.range_test(train_dataloader, end_lr=10, num_iter=100)
    lr_finder.plot(save_path)

    # Find the learning rate with the steepest negative gradient
    losses = history['loss']
    lrs = history['lr']
    min_grad = float('inf')
    best_lr = lrs[0]

    for i in range(1, len(lrs)):
        grad = (losses[i] - losses[i - 1]) / (lrs[i] - lrs[i - 1])
        if grad < min_grad:
            min_grad = grad
            best_lr = lrs[i]

    return best_lr
