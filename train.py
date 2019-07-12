#!/usr/bin/env python

import argparse
from torch import nn
from torch import optim

from fc_model import build_model, load_checkpoint, validate, train
from helper import load_data

parser = argparse.ArgumentParser()

parser.add_argument("data_directory", help="path to folder containing training and validation data")
parser.add_argument("--save_dir",
                    help="set directory to save checkpoints")
parser.add_argument("--arch", help="choose architecture", default='vgg13')
parser.add_argument("--learning_rate",
                    help="Set the learning rate for the optimization step",
                    default=0.01,
                    type=float)
parser.add_argument("--hidden_units",
                    help="Set number of hidden_units",
                    default=512,
                    type=int)
parser.add_argument("--epochs",
                    help="Set number of epochs to train for",
                    default=20,
                    type=int)
parser.add_argument("--gpu",
                    help="Use GPU for inference",
                    action="store_true")

args = parser.parse_args()

device = "cuda" if args.gpu else "cpu"

trainloader, validloader, testloader, class_to_idx = load_data(args.data_directory)

model = build_model(arch=args.arch, hidden_units=args.hidden_units)
model.class_to_idx = class_to_idx
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
train(model, criterion, optimizer, trainloader, validloader, epochs=args.epochs, device=device, save_dir=args.save_dir)
