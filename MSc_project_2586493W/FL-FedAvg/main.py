import json
import datetime
import os
import logging
import torch
import random
from server import *
from client import *
import model, dataset
import pandas as pd
import wandb



wandb.init(project="FL-epoch-50", name="FL-FedAvg-modified resnet18(SGD)")

	
if __name__ == '__main__':
    # Storage container for drawing images
    accs = []  # Storage accuracy
    losses = []  # Storage loss

    # Load Configure File
    with open('utils/conf.json', 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = dataset.get_dataset('./data/', conf['dataset'])  # Obtain training data and test data
    server = Server(conf, eval_datasets)  # create server
    clients = []  # list of clients
    for c in range(conf['n_client']):  # create clients
        clients.append(Client(conf, server.global_model, train_datasets, c))

    for e in range(conf['global_epochs']):  # global epoch
        candidates = random.sample(clients, conf['chosen_client'])  # Randomly select  clients
        weight_accumulator = {}  # Create a calculated parameter dictionary whose value is the sum of the changes calculated by the local model
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)  # Initialize the above parameter dictionary with the same size as the global model

        for c in candidates:  # Obtain and accumulate the changes of local model update one by one
            diff = c.local_train(server.global_model)  # Perform local training and calculate the difference dictionary
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)  # Model aggregation
        acc, loss = server.model_eval()  # Conduct global model test
        accs.append(acc)
        losses.append(loss)
        print('Global model：Epoch{} has finished. Test_accuracy：{:.2f} loss: {:.2f}'.format(e, acc, loss))
        wandb.config.n_client = 50
        wandb.config.chosen_client = 10
        wandb.config.global_epochs = 50
        wandb.config.local_epochs = 10
        wandb.config.batch_size = 32
        wandb.config.lr = 0.01
        wandb.config.momentum = 0.9
        wandb.log({"loss": loss, "test_acc": acc})




		
		
