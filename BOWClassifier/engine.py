import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from colorama import Back, Fore
from utils import accuracy_fn, checkpoint

torch.manual_seed(42)
torch.manual_seed(42)
torch.manual_seed(42)


def run(model: nn.Module,
        device: str,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        accuracy_fn=accuracy_fn,
        epochs=None,
        root=None,
        Exp_ID=None,
        config=None,
        ):

    # Define Buffer for Loss and Accuracy
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    # Train Model
    model.to(device)
    model.train()

    loss_t, acc_t = 0, 0
    for epoch in range(epochs):
        print(Back.WHITE + Fore.BLACK + f"____________ Epoch: {epoch + 1} ____________")
        print(Back.RESET + Fore.BLUE + Exp_ID)
        for batch_counter, (input_data, label) in enumerate(train_dataloader):
            input_data, label = input_data.to(device), label.to(device)

            output = model(input_data.type(torch.float))

            loss_batch = loss_fn(output.squeeze(), label)
            loss_t += loss_batch

            acc_batch = accuracy_fn(outputs=output.squeeze(), target=label)
            acc_t += acc_batch

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        loss_t /= len(train_dataloader)
        acc_t /= len(train_dataloader)

        train_loss.append(loss_t.item())
        train_acc.append(acc_t.item())

        # Validation
        model.eval()
        loss_v, acc_v = 0, 0
        with torch.inference_mode():
            for input_data, label in valid_dataloader:
                output = model(input_data.type(torch.float))

                loss_batch = loss_fn(output.squeeze(), label)
                loss_v += loss_batch

                acc_batch = accuracy_fn(outputs=output.squeeze(), target=label)
                acc_v += acc_batch

            loss_v /= len(valid_dataloader)
            acc_v /= len(valid_dataloader)

            valid_loss.append(loss_v.item())
            valid_acc.append(acc_v.item())

        scheduler.step()

        print(Back.RESET + Fore.GREEN + f"Train Accuracy: {acc_t}")
        print(Back.RESET + Fore.GREEN + f"Validation Accuracy: {acc_v} \n")

        print(Back.RESET + Fore.YELLOW + f"Train Loss: {loss_t}")
        print(Back.RESET + Fore.YELLOW + f"Validation Loss: {loss_v}")

        checkpoint(train_loss=train_loss, train_acc=train_acc,
                   valid_loss=valid_loss, valid_acc=valid_acc,
                   root=root, Exp_ID=Exp_ID,
                   state_dict=model.state_dict(), epoch=epoch,
                   plot_metrics=True, save_state=True,
                   config=config)




