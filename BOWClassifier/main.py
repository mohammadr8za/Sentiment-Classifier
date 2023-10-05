import torch
from torch import nn
from models import models
from data_preparation import data_preparation
from engine import run
from utils import accuracy_fn

torch.manual_seed(42)
torch.manual_seed(42)
torch.manual_seed(42)

# Define root directory of the project
root = r"D:\mreza\TestProjects\Python\NLP\BOW_Classifier"

# Initialization
Epochs = 20
Learning_Rate = [1e-05, 1e-06]
Batch_Size = [8, 16, 32]
Gamma = [0.995]
Lambda = [9e-06]
Dataset_ID = ["IMDB"]
Dropout = [0.06]

# Device Diagnostic Code
device = "cuda" if torch.cuda.is_available() else "cpu"

config = {"Dataset": None,
          "BS": None,
          "LR": None,
          "G": None,
          "L": None,
          "Dropout": None,
          "Epochs": Epochs}

for dataset in Dataset_ID:
    config["Dataset"] = dataset

    for bs in Batch_Size:
        config["BS"] = bs

        for lr in Learning_Rate:
            config["LR"] = lr

            for d in Dropout:
                config["Dropout"] = d

                for g in Gamma:
                    config["G"] = g

                    for lamb in Lambda:
                        config["L"] = lamb

                        Exp_ID = f"{config['Dataset']}_BS{config['BS']}_LR{config['LR']}_D{config['Dropout']}_" \
                                 f"G{config['G']}_L{config['L']}_E{Epochs}"

                        train_dataloader, valid_dataloader, classes_name = data_preparation(root=root,
                                                                                            batch_size=config["BS"])

                        input_sample, _ = next(iter(train_dataloader))

                        model = models.SimpleLinearModel(input_dim=input_sample.shape[-1], hidden_size_1=256,
                                                         hidden_size_2=64, num_classes=2, dropout=config["Dropout"])

                        optimizer = torch.optim.SGD(params=model.parameters(),
                                                    lr=config["LR"],
                                                    weight_decay=config["G"])

                        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=config["G"],
                                                                       step_size=10)

                        loss_fn = nn.CrossEntropyLoss()

                        run(model=model,
                            device=device,
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader,
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn,
                            epochs=Epochs,
                            root=root,
                            Exp_ID=Exp_ID,
                            config=config)
