import os

import torch
from torch.utils.data import random_split
import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )   
    
    print("Using device:", device)
    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    data_path = "train.csv"

    print("Loading data...")
    data = StartingDataset(data_path, device)
    
    train_size = int(0.95 * len(data)) # 95% of data for training
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data,[train_size, val_size])
    
    print("Loading model...")
    model = StartingNetwork().to(device)
    
    print("Training model...")
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
