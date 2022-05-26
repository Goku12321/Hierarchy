# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from hier_tree.tree import Tree, CIFFAR100_HIER_CLASSES
from anytree import LevelGroupOrderIter, PreOrderIter

CIFFAR100_hier_classes = CIFFAR100_HIER_CLASSES()
CIFFAR100_tree = Tree(CIFFAR100_hier_classes.cifar100_dict)

def hier_loss(outputs, labels):
    nls = torch.zeros(outputs.size()).cuda()  # nls: Negative Log Softmax
    if args.gpu:
        nls = nls.cuda()
    for i, l in enumerate(labels):
        for j in CIFFAR100_tree.tree_path[l.item()]:
            nls[i:i + 1, CIFFAR100_tree.siblings[j]] = -log_softmax(outputs[i:i + 1, CIFFAR100_tree.siblings[j]])
    hot_vector = CIFFAR100_tree.one_hot_vector(labels)
    if args.gpu:
        hot_vector = hot_vector.cuda()
    return (hot_vector * nls).sum()

def train(epoch):
    start = time.time()
    net.train()

    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.hier:
            labels = torch.LongTensor([mapping_indices[label] for label in labels])

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)

        if args.hier:
            # nls = torch.zeros(outputs.size()).cuda() #nls: Negative Log Softmax
            # if args.gpu:
            #     nls = nls.cuda()
            # for i, l in enumerate(labels):
            #     for j in CIFFAR100_tree.tree_path[l.item()]:
            #         nls[i:i+1, CIFFAR100_tree.siblings[j]] = -log_softmax(outputs[i:i+1, CIFFAR100_tree.siblings[j]])
            # hot_vector = CIFFAR100_tree.one_hot_vector(labels)
            # if args.gpu:
            #     hot_vector = hot_vector.cuda()
            # loss = (hot_vector * nls).sum()
            loss = hier_loss(outputs, labels)
        else:
            loss = cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)


@torch.no_grad()
def eval_training(epoch=0, hier=False, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = cross_entropy(outputs, labels)

        test_loss += loss.item()

        if args.hier:
            level_indices = []
            for i, level in enumerate(LevelGroupOrderIter(CIFFAR100_tree.root)):
                if i == 0:
                    continue
                level_indices.append([node.index for node in level])

            # softmax to all siblings groups
            probs = torch.zeros(outputs.size()).cuda()
            for s in CIFFAR100_tree.siblings_groups:
                probs[:, s] = softmax(outputs[:, s])

            CIFFAR100_tree.root.prob = torch.ones([outputs.size(0), 1]).cuda()
            for i, node in enumerate(PreOrderIter(CIFFAR100_tree.root)):
                if i == 0: # first node doesn't have parent
                    continue
                parent = node.ancestors[-1]
                node.prob = (probs[:, node.index].unsqueeze(1) * parent.prob).cuda()

            hier_labels = CIFFAR100_tree.make_hier_labels(labels)
            # for each level get the max
            for indices in level_indices:
                print(probs[:, indices].size())
                _, preds = probs[:, indices].max(1)
                print(torch.tensor(indices)[preds])
        else:
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)


@torch.no_grad()
def eval_training_hier(epoch=0, hier=False, tb=True):
    start = time.time()
    net.eval()


    test_loss = 0.0  # cost function error
    correct_per_level = torch.zeros(len(CIFFAR100_tree.level_indices))

    for (images, labels) in cifar100_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = torch.LongTensor([mapping_indices[label] for label in labels]) .cuda()

        outputs = net(images)
        loss = hier_loss(outputs, labels)

        test_loss += loss.item()

        # softmax to all siblings groups
        probs_cond = torch.zeros(outputs.size()).cuda()
        probs = torch.zeros(outputs.size()).cuda()
        for s in CIFFAR100_tree.siblings_groups:
            probs_cond[:, s] = softmax(outputs[:, s])

        CIFFAR100_tree.root.prob = torch.ones([outputs.size(0), 1]).cuda()
        for i, node in enumerate(PreOrderIter(CIFFAR100_tree.root)):
            if i == 0:  # first node doesn't have parent
                continue
            parent = node.ancestors[-1]
            node.prob = (probs_cond[:, node.index].unsqueeze(1) * parent.prob).cuda()
            probs[:, node.index] = node.prob.squeeze()

        hier_labels = CIFFAR100_tree.make_hier_labels(labels)
        # for each level get the max
        for i,indices in enumerate(CIFFAR100_tree.level_indices):
            # print(probs[:, indices].size())
            _, preds = probs[:, indices].max(1)
            preds = torch.tensor(indices)[preds]
            correct_per_level[i] += preds.eq(hier_labels[:,i]).sum()


    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print(f'Test set: Epoch: {epoch}, Average loss: {len(cifar100_test_loader.dataset):.4f}', end=', ')
    for i, corrects in enumerate(correct_per_level):
        print(f'Accuracy Level {i+1}: {corrects.item()/len(cifar100_test_loader.dataset):.4f}', end=', ')


    print(f'Time consumed:{finish - start:.2f}s')

    # add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        for i,corrects in enumerate(correct_per_level):
            writer.add_scalar(f'Test/Accuracy Level {i+1}', corrects.item()/len(cifar100_test_loader.dataset), epoch)


    return correct_per_level[-1].item() / len(cifar100_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-hier', action='store_true', default=False, help='hierarchy classification or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-workers', type=int, default=0, help='0 for debug, 4 for running is recommended')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.workers,
        batch_size=args.b,
        shuffle=True
    )


    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.workers,
        batch_size=args.b,
        shuffle=True
    )

    if args.hier:
        # create labels mapping
        mapping_indices = [-1] * 120
        for i, cls in enumerate(cifar100_training_loader.dataset.class_to_idx):
            mapping_indices[i] = CIFFAR100_tree.get_node_by_name(cls).index

    cross_entropy = nn.CrossEntropyLoss()
    cross_entropy_no_red = nn.CrossEntropyLoss(reduction='none')
    softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)
    nlll = nn.NLLLoss(reduction='none')
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        # ----------------------- train epoch -----------------------
        train(epoch)
        acc = eval_training_hier(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
