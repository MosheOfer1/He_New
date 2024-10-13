import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
import copy


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

            loss = self._train_batch(batch)

            # Compute the smoothed loss
            avg_loss = self.smooth(loss.item())
            self.history["lr"].append(lr_scheduler.get_lr()[0])
            self.history["loss"].append(avg_loss)

            if iteration > 0 and self.history["loss"][-1] > self.diverge_th * self.best_loss:
                print("Stopping early, the loss has diverged")
                break

            if avg_loss < self.best_loss or self.best_loss is None:
                self.best_loss = avg_loss

            lr_scheduler.step()

        return self.history

    def _train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._forward_pass(batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def _forward_pass(self, batch):
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**inputs)
        loss = self.criterion(outputs.logits, inputs['input_ids_3'])
        return loss

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


def find_best_lr(model, train_dataloader, criterion, device, save_path):
    model_copy = copy.deepcopy(model)
    optimizer = Adam(model_copy.parameters(), lr=1e-7, weight_decay=1e-2)
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
