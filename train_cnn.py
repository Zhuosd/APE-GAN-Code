# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms

from tqdm import tqdm

from models import MnistCNN, CifarCNN
from utils import accuracy, fgsm


def load_dataset(args):
    if args.data == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root = './', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root = './', train=False, download=False,
                           transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=128, shuffle=False)
    elif args.data == "cifar":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])),
            batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root = './', train=False, download=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])),
            batch_size=128, shuffle=False)
    return train_loader, test_loader


def load_cnn(args):
    if args.data == "mnist":
        return MnistCNN
    elif args.data == "cifar":
        return CifarCNN


def main(args):
    print("Generating Model ...")
    print("-" * 30)

    train_loader, test_loader = load_dataset(args)
    CNN = load_cnn(args)
    model = CNN().cuda()
    cudnn.benchmark = True

    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    scheduler = lr_scheduler.MultiStepLR(opt, milestones=args.milestones, gamma=args.gamma)
    loss_func = nn.CrossEntropyLoss().cuda()

    epochs = args.epochs
    print_str = "\t".join(["{}"] + ["{:.6f}"] * 4)
    print("\t".join(["{:}"] * 5).format("Epoch", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."))
    for e in range(epochs):
        train_loss, train_acc, train_n = 0, 0, 0
        test_loss, test_acc, test_n = 0, 0, 0

        model.train()
        for x, t in tqdm(train_loader, total=len(train_loader), leave=False):
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * t.size(0)
            train_acc += accuracy(y, t)
            train_n += t.size(0)

        model.eval()
        for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)

            test_loss += loss.item() * t.size(0)
            test_acc += accuracy(y, t)
            test_n += t.size(0)
        scheduler.step()
        print(print_str.format(e, train_loss / train_n, test_loss / test_n,
                               train_acc / train_n * 100, test_acc / test_n * 100))
    torch.save({"state_dict": model.state_dict()}, "cnn.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--milestones", type=list, default=[50, 75])
    parser.add_argument("--gamma", type=float, default=0.1)
    args = parser.parse_args()
    main(args)
