from transformers import PreTrainedModel
from typing import Dict, Callable, Iterable
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

Real = Prediction = Iterable[float]

class BaseTransformerModel(nn.Module):

    def __init__(self, transformer: PreTrainedModel,
                       metrics: Dict[str, Callable[[Real, Prediction], float]],
                       freeze_layers: bool = False):

        super(BaseTransformerModel, self).__init__()

        self.transformer_encoder = transformer
        self.metrics = metrics

        if freeze_layers:
            self.freeze_layers()
    
    def freeze_layers(self) -> None:

        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_layers(self) -> None:

        for param in self.transformer_encoder.parameters():
            param.requires_grad = True

    def forward(self, input_ids, att_masks):

        raise NotImplementedError("Forward not implemented.")

    def evaluate(self, eval_dl: DataLoader, 
                        criterion: torch.nn,
                        cuda: bool = False,
                        device: torch.device = torch.device("cpu:0")) -> None:
        # evaluate
        self.eval()

        with torch.no_grad():

            preds = []
            real = []
            batch_losses = []

            for input_batch in eval_dl:

                if cuda:
                    input_batch = [x.cuda(device) for x in input_batch]

                input_ids, att_masks, labels = input_batch

                outputs = self(input_ids, att_masks) 
                loss = criterion(outputs.squeeze(), labels)

                outputs = F.softmax(outputs, dim=1)
                outputs = outputs.argmax(axis=1)
                
                preds.extend(outputs.tolist())
                real.extend(labels.tolist())
                batch_losses.append(loss.item())

            results = {}
            for metric_name, metric in self.metrics.items():
                results[metric_name] = metric(real, preds)

            mean_loss = np.mean(batch_losses)

            tqdm.write(f"\ttrain_loss: {self.last_train_loss} // test_loss: {mean_loss}// metrics: {str(results)}\n")

        return preds, mean_loss

    def fit(self, epochs: int, 
                    train_dl: DataLoader, 
                    test_dl: DataLoader,
                    criterion: torch.nn,
                    optimizer: torch.optim,
                    scheduler: torch.optim.lr_scheduler = None,
                    cuda: bool = False, 
                    device: torch.device = torch.device("cpu:0")):

        train_losses = []
        eval_losses = []

        for epoch in range(epochs):
            # train
            self.train()
            batch_losses = []

            batches = len(train_dl)

            for batch_input in tqdm(train_dl, total=batches, desc="- Remaining batches"):

                if cuda:
                    batch_input = [x.cuda(device) for x in batch_input]

                input_ids, att_masks, labels = batch_input

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(input_ids, att_masks)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                
                optimizer.step()

                if scheduler is not None: scheduler.step()

                batch_losses.append(loss.item())
        
            train_loss = np.mean(batch_losses)
            self.last_train_loss = train_loss

            # evaluate
            tqdm.write(f"Epoch: {epoch+1}")
            _, eval_loss = self.evaluate(test_dl, criterion, cuda, device)

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

        return train_losses, eval_losses