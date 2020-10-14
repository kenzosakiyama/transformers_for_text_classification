from torch_lr_finder import LRFinder, TrainDataLoaderIter
import matplotlib.pyplot as plt
from typing import Union
import torch
import torch.nn as nn

try:
    from apex import amp

    IS_AMP_AVAILABLE = True
except ImportError:
    IS_AMP_AVAILABLE = False

class TextDataLoaderIter(TrainDataLoaderIter):

    def __init__(self, dl):
        super().__init__(dl, True)

    def inputs_labels_from_batch(self, batch_data):

        input_ids, att_masks, labels = batch_data 

        # returning Tuple[inputs, labels]
        return (input_ids, att_masks), labels

class TransformerLRFinder(LRFinder):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
        self.model.train()
        total_loss = None  # for late initialization

        self.optimizer.zero_grad()
        for i in range(accumulation_steps):

            # Dealing with hugging face transformer inputs
            inputs, labels = next(train_iter)
            input_ids, att_masks = inputs

            input_ids = input_ids.to(self.device)
            att_masks = att_masks.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(input_ids, att_masks)
            loss = self.criterion(outputs, labels)

            # Loss should be averaged in each step
            loss /= accumulation_steps

            # Backward pass
            if IS_AMP_AVAILABLE and hasattr(self.optimizer, "_amp_stash"):
                # For minor performance optimization, see also:
                # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = ((i + 1) % accumulation_steps) != 0

                with amp.scale_loss(
                    loss, self.optimizer, delay_unscale=delay_unscale
                ) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        self.optimizer.step()

        return total_loss.item()

def run_lr_finder(train_dl: torch.utils.data.DataLoader,
                  model: nn.Module,
                  optimizer: torch.optim,
                  criterion: nn.Module,
                  device: Union[str, torch.device] = "cpu",
                  end_lr: int = 10,
                  num_iter: int = 100,
                  save_plot: bool = False
                  ) -> None:

    lr_finder = TransformerLRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(TextDataLoaderIter(train_dl), end_lr=end_lr, num_iter=num_iter)
    lr_finder.plot()
    if save_plot: plt.savefig("lr_finder.png")
    lr_finder.reset()
