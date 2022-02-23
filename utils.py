from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as TF
import numpy as np
import torch
from torch import nn
import time
import os

from pars import PARS
from setup_net import setup_net
from loss import *


def get_BYOL_transforms(is_first=True):
    transforms = torch.nn.Sequential(
        TF.RandomResizedCrop(32),
        TF.RandomHorizontalFlip(p=0.5),
        TF.RandomApply(torch.nn.ModuleList([
            TF.ColorJitter(brightness=0.4, contrast=0.4,
                           saturation=0.2, hue=0.1)
        ]), p=0.8),
        TF.RandomApply(torch.nn.ModuleList([
            TF.Grayscale(num_output_channels=3)
        ]), p=0.2),
        # TF.RandomApply(torch.nn.ModuleList([
        #     TF.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        # ]), p=1.0 if is_first else 0.1),
        TF.RandomSolarize(threshold=0.5, p=0.0 if is_first else 0.2)
    )
    scripted_transforms = torch.jit.script(transforms)
    return scripted_transforms

def get_data(datapath, dataset, num_train):
    if dataset == 'Cifar100':
        trainset = dset.CIFAR100(root=datapath, train=True, download=True)
        train_dat = (trainset.data.transpose(0, 3, 1, 2) / 255 - 0.5) / 0.5
        train_tar = np.array(trainset.targets)
    elif dataset == 'Cifar10':
        trainset = dset.CIFAR10(root=datapath, train=True, download=True)
        train_dat = (trainset.data.transpose(0, 3, 1, 2) / 255 - 0.5) / 0.5
        train_tar = np.array(trainset.targets)
    else:
        trainpath = datapath + 'ImageNet_train'
        train_tar_path = datapath + 'ImageNet_train_tar'
        with open(trainpath, 'rb') as f:
            train_dat = np.load(f)[:int(1.1*num_train)]
            train_dat = (train_dat / 255 - 0.5) / 0.5
        with open(train_tar_path, 'rb') as h:
            train_tar = np.load(h)[:int(1.1*num_train)]

    if dataset == 'Cifar100':
        testset = dset.CIFAR100(root=datapath, train=False, download=True)
        test_dat = (testset.data.transpose(0, 3, 1, 2) / 255 - 0.5) / 0.5
        test_tar = np.array(testset.targets)
    elif dataset == 'Cifar10':
        testset = dset.CIFAR10(root=datapath, train=False, download=True)
        test_dat = (testset.data.transpose(0, 3, 1, 2) / 255 - 0.5) / 0.5
        test_tar = np.array(testset.targets)
    else:
        testpath = datapath + 'ImageNet_test'
        test_tar_path = datapath + 'ImageNet_test_tar'
        with open(testpath, 'rb') as ft:
            test_dat = np.load(ft)
        with open(test_tar_path, 'rb') as ht:
            test_tar = np.load(ht)
        test_dat = (test_dat/255-0.5)/0.5

    return train_dat[:num_train], train_tar[:num_train], train_dat[num_train:], train_tar[num_train:], test_dat, test_tar


def train_model(data, fix, model, pars, ep_loss, ep_acc, criterion=None, optimizer = None):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    device = pars.device
    dtype = torch.float32
    train_dat = data[0]
    train_tar = data[1]
    val_dat = data[2]
    val_tar = data[3]

    print(fix)
    print(model)

    fix = fix.to(device=device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    if pars.train_unsupervised:
        lr = pars.LR
        if not criterion:
            criterion = SimCLRLoss(pars.batch_size, pars.device)
        params = list(fix.parameters())+list(model.parameters())

    else:
        lr = pars.clf_lr
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()
        params = model.parameters()
    print(criterion)

    if not optimizer:
        optimizer = torch.optim.Adam(params, lr=lr)
    print(optimizer)

    if (pars.unsupervised) and (not pars.train_unsupervised):
        n_epochs = pars.clf_epochs
    else:
        n_epochs = pars.epochs

    with torch.autograd.set_detect_anomaly(True):
        for e in range(n_epochs):
            running_loss = 0
            start_time = time.time()

            for j in np.arange(0, len(train_tar), pars.batch_size):
                model.train()  # put model to training mode
                # move to device, e.g. GPU
                x = torch.from_numpy(
                    train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)
                if pars.train_unsupervised:
                    img = x * 0.5 + 0.5

                    transform1 = get_BYOL_transforms(True)
                    transform2 = get_BYOL_transforms(False)
                    x1 = transform1(img)
                    x2 = transform2(img)
                    
                    x = torch.cat((x1, x2), dim=0)

                else:
                    y = torch.from_numpy(
                        train_tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)

                with torch.no_grad():
                    x1 = fix(x)
                scores = model(x1)

                if pars.train_unsupervised:
                    loss = criterion(scores)
                else:
                    loss = criterion(scores, y)

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_time = time.time()-start_time
            running_loss /= (len(train_tar)/pars.batch_size)
            ep_loss.append(running_loss)
            if pars.train_unsupervised:
                print('Epoch %d, loss = %.4f, time: %0.4f' %
                      (e, running_loss, end_time))
            else:
                acc = check_accuracy(val_dat, val_tar, fix, model, pars)
                print('Epoch %d, loss = %.4f, val.acc = %.4f' %
                      (e, running_loss, acc))
                ep_acc.append(acc)

            # expdir =pars.savepath + "checkpoint/"
            # if not os.path.exists(expdir):
            #     os.makedirs(expdir)
            # torch.save(
            #     model.state_dict(), expdir +'epochs_{}.pt'.format(e)
            # )


def check_accuracy(dat, tar, fix, model, pars):

    device = pars.device

    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for j in np.arange(0, len(tar), pars.batch_size):
            # move to device, e.g. GPU
            x = torch.from_numpy(
                dat[j:j+pars.batch_size]).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(
                tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
            x1 = fix(x)
            scores = model(x1)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        #print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def train_unsupervised(pars, criterion=None, clf_criterion=None, optimizer=None):
    print(pars)

    expdir = pars.savepath+pars.architecture+"/"+pars.loss+"/"
    EXP_NAME = '{}_{}_{}_LR_{}_Epochs_{}_CLF_{}_{}_LR_{}_Epochs_{}'.format(pars.nonlinear, pars.dataset, pars.OPT, pars.LR, pars.epochs,
        pars.clf_dataset, pars.clf_opt, pars.clf_lr, pars.clf_epochs)
    if pars.loss == 'BarlowTwins':
        EXP_NAME += '_lambda_'+str(pars.BTlambda)
    
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    print(expdir)
    print(EXP_NAME)

    pars.train_unsupervised = pars.unsupervised

    dtype = torch.float32

    data = get_data(pars.datapath, pars.dataset, num_train=500000)
    clf_data = get_data(pars.datapath, pars.clf_dataset, num_train=45000)

    test_acc_all = []
    for rep in range(pars.repeat):
        print("\nRep {}".format(rep+1))

        net, classifier, head = setup_net(pars)
        if pars.loadnet:
            net.load_state_dict(pars.loadnet)
        if pars.loadclf:
            classifier.load_state_dict(pars.loadclf)
        print(net)
        print(classifier)
        
        print(head)

        val_loss = []
        val_acc = []
        lw_test_acc = []

        head_loss = []

        fix = nn.Sequential()
        model = nn.Sequential(
            net,
            head)
        pars.train_unsupervised = True
 
        train_model(data, fix, model, pars, head_loss, None, criterion, optimizer)

        print('Train Classifier')
        pars.train_unsupervised = False
        print(net)
        print(classifier)
        train_model(clf_data, net, classifier, pars, val_loss, val_acc, clf_criterion, optimizer)
        test_acc = check_accuracy(
            clf_data[4], clf_data[5], net, classifier, pars)
        print('Rep: %d, te.acc = %.4f' % (rep+1, test_acc))
        lw_test_acc.append(test_acc)
        

        np.save(expdir+'head_loss_rep_{}_'.format(rep+1) +
                EXP_NAME, head_loss)
        np.save(expdir+'loss_rep_{}_'.format(rep+1) + EXP_NAME, val_loss)
        np.save(expdir+'val.acc_rep_{}_'.format(rep+1) + EXP_NAME, val_acc)
        np.save(expdir+'te.acc_rep_{}_'.format(rep+1) + EXP_NAME, lw_test_acc)

        if pars.epochs:
            torch.save(net.state_dict(), expdir +
                       'net_rep_{}_'.format(rep+1) + EXP_NAME + '.pt')
        torch.save(classifier.state_dict(), expdir +
                   'clf_rep_{}_'.format(rep+1) + EXP_NAME + '.pt', )

        test_acc_all.append(lw_test_acc)

    print('\nAll reps test.acc:')
    for acc in test_acc_all:
        print(acc)
    np.save(expdir+'te.acc.all_' + EXP_NAME, test_acc_all)
