import os
from colorama import Fore
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from os.path import join

torch.manual_seed(42)
torch.manual_seed(42)
torch.manual_seed(42)


def accuracy_fn(outputs, target):

    correct_count = torch.sum(outputs.argmax(dim=1) == target)

    return (correct_count / outputs.shape[0]).detach().numpy()


def checkpoint(train_loss, train_acc, valid_loss, valid_acc, root, Exp_ID,
               state_dict, epoch, plot_metrics=True, save_state=False, config=None):
    # Plot Metrics and Save Model Weights

    save_root = join(root, "Experiments", "train", Exp_ID)

    if plot_metrics:
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].plot(train_loss, label="train loss")
        ax[0].plot(valid_loss, label="valid loss")
        ax[0].set_title("loss")

        ax[1].plot(train_acc, label="train acc")
        ax[1].plot(valid_acc, label="valid acc")
        ax[1].set_title("accuracy")

        fig.suptitle("METRICS VISUALIZATION")
        fig.tight_layout()

        os.makedirs(join(save_root, "metrics"), exist_ok=True)
        plt.savefig(join(save_root, "metrics", "metrics_plot.png"))

    if save_state:
        os.makedirs(join(save_root, "states"), exist_ok=True)
        torch.save(obj=state_dict, f=join(save_root, "states", "epoch_" + f"{epoch}" + ".pt"))

    if epoch == 0:
        config_df = pd.DataFrame(config, index=[0])
        config_df.to_csv(join(save_root, "config.csv"), index=False)

    print(Fore.RESET + "Checkpoint !")

