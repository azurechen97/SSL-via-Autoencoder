import torchvision.datasets as dset
import torchvision.transforms as TF
import numpy as np
import torch
from torch import nn
import time
import os

from pars import PARS
from setup_net import *
from loss import *

import matplotlib.pyplot as plt

import sklearn

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


def train_model(data, fix, model, pars, ep_loss, ep_acc, criterion=None, optimizer = None, vis=None):

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
            if pars.loss == "BarlowTwins":
                criterion = BarlowTwinsLoss(
                    pars.batch_size, pars.lam, pars.device)
            else:
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

    if pars.train_unsupervised:
        n_epochs = pars.epochs
    else:
        n_epochs = pars.clf_epochs

    ep_lr = []
    if not pars.train_unsupervised:
        ep_val_loss = []

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
                    x_new = fix(x)
                scores = model(x_new)

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
            ep_lr.append(optimizer.param_groups[0]['lr'])

            if not pars.train_unsupervised:
                val_running_loss = 0
                with torch.no_grad():
                    for j in np.arange(0, len(val_tar), pars.batch_size):
                        # move to device, e.g. GPU
                        val_x = torch.from_numpy(
                            val_dat[j:j+pars.batch_size]).to(device=device, dtype=torch.float32)
                        val_y = torch.from_numpy(
                            val_tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
                        val_x_new = fix(val_x)
                        val_scores = model(val_x_new)
                        val_loss = criterion(val_scores, val_y)
                        val_running_loss += val_loss.item()

                    val_running_loss /= (len(val_tar)/pars.batch_size)
                    ep_val_loss.append(val_running_loss)
            
            if pars.train_unsupervised:
                print('Epoch %d, loss = %.4f, time: %0.4f' %
                      (e, running_loss, end_time))
            else:
                acc = check_accuracy(val_dat, val_tar, fix, model, pars)
                print('Epoch %d, loss = %.4f, val.loss = %.4f, val.acc = %.4f' %
                      (e, running_loss, val_running_loss, acc))
                ep_acc.append(acc)
            
            if vis is not None and e % 5 == 4:
                if pars.train_unsupervised:
                    s = ""
                else:
                    s = "clf "
                    vis.line(np.array(ep_acc), np.arange(len(ep_acc)),  win="val acc", name="acc",
                             opts=dict(title="val acc",
                                       xlabel="epochs",
                                       ylabel="acc"))

                vis.line(np.array(ep_loss), np.arange(len(ep_loss)),  win=s+"loss", name="loss",
                        opts=dict(title=s+"loss",
                                xlabel="epochs",
                                ylabel="loss"))
                if not pars.train_unsupervised:
                    vis.line(np.array(ep_val_loss), np.arange(len(ep_val_loss)),  win=s+"loss", update='append', name="val loss")
                vis.line(np.array(ep_loss), np.arange(len(ep_loss)),  win=s+"loss (log)", name="loss",
                        opts=dict(title=s+"loss (log)",
                                xlabel="epochs",
                                ylabel="loss",
                                ytype="log"))
                if not pars.train_unsupervised:
                    vis.line(np.array(ep_val_loss), np.arange(
                        len(ep_val_loss)),  win=s+"loss (log)", update='append', name="val loss")


            # expdir =pars.savepath + "checkpoint/"
            # if not os.path.exists(expdir):
            #     os.makedirs(expdir)
            # torch.save(
            #     model.state_dict(), expdir +'epochs_{}.pt'.format(e)
            # )

def train_model_ae(data, fix, model, decoder, pars, ep_loss, criterion_re=None, criterion_sim=None, optimizer = None, vis=None):

    device = pars.device
    dtype = torch.float32
    train_dat = data[0]
    train_tar = data[1]

    print(fix)
    print(model)
    print(decoder)

    fix = fix.to(device=device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    decoder = decoder.to(device=device)

    lr = pars.LR
    if not criterion_re:
        criterion_re = nn.MSELoss()
    if not criterion_sim:
        criterion_sim = TwinMSELoss(pars.batch_size, pars.device)
    params = list(fix.parameters())+list(model.parameters())+list(decoder.parameters())
    if pars.lam == -1:
        criterion_mtl = MultiTaskLoss(pars.batch_size, pars.device)
        params += list(criterion_mtl.parameters())

    print(criterion_re)
    print(criterion_sim)

    if not optimizer:
        optimizer = torch.optim.Adam(params, lr=lr)
    print(optimizer)

    n_epochs = pars.epochs

    ep_loss_re = []
    ep_loss_sim = []
    ep_lr = []

    with torch.autograd.set_detect_anomaly(True):
        for e in range(n_epochs):
            running_loss = 0
            running_loss_re = 0
            running_loss_sim = 0
            start_time = time.time()

            for j in np.arange(0, len(train_tar), pars.batch_size):
                model.train()  # put model to training mode
                decoder.train()
                if pars.lam == -1:
                    criterion_mtl.train()

                # move to device, e.g. GPU
                x = torch.from_numpy(
                    train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)

                img = x * 0.5 + 0.5

                transform1 = get_BYOL_transforms(True)
                transform2 = get_BYOL_transforms(False)
                x1 = transform1(img)
                x2 = transform2(img)
                
                x = torch.cat((x1, x2), dim=0)

                with torch.no_grad():
                    x_new = fix(x)
                scores = model(x_new)
                x_re = decoder(scores)

                loss_re = criterion_re(x_re,x_new)
                loss_sim = criterion_sim(scores)
                
                if pars.lam == -1: # multi-task loss
                    loss = criterion_mtl(loss_re, loss_sim)
                else:
                    # loss= (1-pars.lam)*loss_sim+pars.lam*loss_re
                    loss= loss_sim+pars.lam*loss_re
                # loss = loss_sim

                running_loss += loss.item()
                running_loss_re += loss_re.item()
                running_loss_sim += loss_sim.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_time = time.time()-start_time
            running_loss /= (len(train_tar)/pars.batch_size)
            running_loss_re /= (len(train_tar)/pars.batch_size)
            running_loss_sim /= (len(train_tar)/pars.batch_size)
            ep_loss.append(running_loss)
            ep_loss_re.append(running_loss_re)
            ep_loss_sim.append(running_loss_sim)

            ep_lr.append(optimizer.param_groups[0]['lr'])

            print('Epoch %d, loss = %.4f, time: %0.4f' %
                    (e, running_loss, end_time))
            print('reconstruction loss = %.4f, similarity loss: %0.4f' %
                    (running_loss_re, running_loss_sim))
            if pars.lam == -1:
                print(list(criterion_mtl.parameters())[0])
            if vis is not None and e % 5 == 4:
                ind = np.random.choice(pars.batch_size)

                vis.line(np.array(ep_loss), np.arange(len(ep_loss)),  win="loss", name="loss",
                        opts=dict(title="loss",
                                xlabel="epochs",
                                ylabel="loss"))
                vis.line(np.array(ep_loss_re), np.arange(len(ep_loss)),  win="loss", update='append', name="reconstruction loss")
                vis.line(np.array(ep_loss_sim), np.arange(len(ep_loss)),  win="loss", update='append', name="similarity loss")

                vis.line(np.array(ep_loss), np.arange(len(ep_loss)),  win="loss (log)", name="loss",
                        opts=dict(title="loss (log)",
                                xlabel="epochs",
                                ylabel="loss",
                                ytype = "log"))
                vis.line(np.array(ep_loss_re), np.arange(len(ep_loss)),  win="loss (log)",
                        update='append', name="reconstruction loss")
                vis.line(np.array(ep_loss_sim), np.arange(len(ep_loss)),
                        win="loss (log)", update='append', name="similarity loss")

                vis.image(img[ind, :, :, :], win="original", opts=dict(
                    caption="original image", store_history=True))
                vis.image(x_new[ind, :, :, :], win="img 1", opts=dict(caption="distorted image 1", store_history=True))
                vis.image(x_re[ind, :, :, :],
                        win="rec 1", opts=dict(caption="reconstructed image 1", store_history=True))
                vis.image(x_new[ind+pars.batch_size, :, :, :], win="img 2", opts=dict(
                    caption="distorted image 2", store_history=True))
                vis.image(x_re[ind+pars.batch_size, :, :, :],
                          win="rec 2", opts=dict(caption="reconstructed image 2", store_history=True))


def train_model_logistic(data, fix, model, pars):

    device = pars.device
    dtype = torch.float32
    train_dat = data[0]
    train_tar = data[1]
    val_dat = data[2]
    val_tar = data[3]

    print(fix)
    print(model)

    fix = fix.to(device=device)
    # model = model.to(device=device)  # move the model parameters to CPU/GPU

    for j in np.arange(0, len(train_tar), pars.batch_size):
        x = torch.from_numpy(
            train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)
        with torch.no_grad():
            x1_tensor = fix(x)
        x1_tensor = x1_tensor.flatten(start_dim=1).cpu()
        if j == 0:
            x1 = x1_tensor.detach().numpy()
        else:
            x1 = np.concatenate([x1, x1_tensor.detach().numpy()])
    y = train_tar

    model.fit(x1, y)
    scores = model.decision_function(x1)

    loss = sklearn.metrics.log_loss(y, scores)

    for j in np.arange(0, len(val_tar), pars.batch_size):
        val_x = torch.from_numpy(
            val_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)
        with torch.no_grad():
            val_x1_tensor = fix(val_x)
        val_x1_tensor = val_x1_tensor.flatten(start_dim=1).cpu()
        if j == 0:
            val_x1 = val_x1_tensor.detach().numpy()
        else:
            val_x1 = np.concatenate([val_x1, val_x1_tensor.detach().numpy()])

    val_pred = model.predict(val_x1)
    acc = sklearn.metrics.accuracy_score(val_tar, val_pred)
    print('loss = %.4f, val.acc = %.4f' % (loss, acc))

    return loss, acc

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
            x_new = fix(x)
            scores = model(x_new)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        #print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def check_accuracy_logistic(dat, tar, fix, model, pars):

    device = pars.device

    for j in np.arange(0, len(tar), pars.batch_size):
        x = torch.from_numpy(
            dat[j:j+pars.batch_size]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            x1_tensor = fix(x)
        x1_tensor = x1_tensor.flatten(start_dim=1).cpu()
        if j == 0:
            x1 = x1_tensor.detach().numpy()
        else:
            x1 = np.concatenate([x1, x1_tensor.detach().numpy()])

    pred = model.predict(x1)
    acc = sklearn.metrics.accuracy_score(tar, pred)

    return acc

def train_unsupervised(pars, criterion=None, clf_criterion=None, sklearn_classifier=None, optimizer=None, vis=None):
    # print(pars)

    expdir = pars.savepath+pars.architecture+"/"+pars.loss+"/"
    EXP_NAME_NET = '{}_{}_{}_LR_{}_Epochs_{}'.format(pars.nonlinear, pars.dataset, pars.OPT, pars.LR, pars.epochs)
    if pars.loss == 'BarlowTwins':
        EXP_NAME_NET += '_lam_'+str(pars.lam)
    EXP_NAME = EXP_NAME_NET+'_CLF_{}_{}_LR_{}_Epochs_{}'.format(
        pars.clf_dataset, pars.clf_opt, pars.clf_lr, pars.clf_epochs)
    
    if not os.path.exists(expdir):
        os.makedirs(expdir)

    print(expdir)
    print(EXP_NAME)

    pars.train_unsupervised = True

    dtype = torch.float32

    data = get_data(pars.datapath, pars.dataset, num_train=500000)
    clf_data = get_data(pars.datapath, pars.clf_dataset, num_train=45000)

    test_acc_all = []
    for rep in range(pars.repeat):
        print("\nRep {}".format(rep+1))

        net, classifier, head = setup_net(pars)

        net_name = expdir + 'net_rep_{}_'.format(rep+1) + EXP_NAME_NET + '.pt'
        clf_name = expdir + 'clf_rep_{}_'.format(rep+1) + EXP_NAME + '.pt'
        if not os.path.isfile(net_name):
            pars.loadnet = False
        if not os.path.isfile(clf_name):
            pars.loadclf = False
        if pars.loadnet:
            net.load_state_dict(torch.load(net_name))
            print(net_name, "loaded")
        if pars.loadclf:
            classifier.load_state_dict(torch.load(clf_name))
            print(clf_name, "loaded")

        # print(net)
        # print(classifier)
        # print(head)

        val_loss = []
        val_acc = []
        lw_test_acc = []

        head_loss = []

        fix = nn.Sequential()
        model = nn.Sequential(
            net,
            head)
        pars.train_unsupervised = True

        if not pars.loadnet:
            print('Train Net')
            train_model(data, fix, model, pars, head_loss, None, criterion, optimizer,vis)

        pars.train_unsupervised = False
        # print(net)
        if not sklearn_classifier:
            # print(classifier)
            if not pars.loadclf:
                print('Train Classifier')
                train_model(clf_data, net, classifier, pars, val_loss, val_acc, clf_criterion, optimizer, vis)
            test_acc = check_accuracy(
                clf_data[4], clf_data[5], net, classifier, pars)
        else:
            print(sklearn_classifier)
            train_model_logistic(clf_data, net, sklearn_classifier, pars)
            test_acc = check_accuracy_logistic(
                clf_data[4], clf_data[5], net, sklearn_classifier, pars)
        print('Rep: %d, te.acc = %.4f' % (rep+1, test_acc))
        lw_test_acc.append(test_acc)
        

        np.save(expdir+'head_loss_rep_{}_'.format(rep+1) +
                EXP_NAME_NET, head_loss)
        np.save(expdir+'loss_rep_{}_'.format(rep+1) + EXP_NAME, val_loss)
        np.save(expdir+'val.acc_rep_{}_'.format(rep+1) + EXP_NAME, val_acc)
        np.save(expdir+'te.acc_rep_{}_'.format(rep+1) + EXP_NAME, lw_test_acc)

        if pars.epochs and not pars.loadnet:
            torch.save(net.state_dict(), net_name)
        if not sklearn_classifier and not pars.loadclf:
            torch.save(classifier.state_dict(), clf_name)

        test_acc_all.append(lw_test_acc)

    print('\nAll reps test.acc:')
    for acc in test_acc_all:
        print(acc)
    np.save(expdir+'te.acc.all_' + EXP_NAME, test_acc_all)


def train_unsupervised_ae(pars, criterion_re=None, criterion_sim=None, clf_criterion=None, sklearn_classifier=None, optimizer=None, vis=None):
    # print(pars)

    expdir = pars.savepath+pars.architecture+"/AE/channel_"+str(pars.decoder_channel)+"/"
    EXP_NAME_NET = '{}_{}_{}_LR_{}_Epochs_{}'.format(
        pars.nonlinear, pars.dataset, pars.OPT, pars.LR, pars.epochs)
    EXP_NAME_NET += '_lam_'+str(pars.lam)
    EXP_NAME = EXP_NAME_NET+'_CLF_{}_{}_LR_{}_Epochs_{}'.format(
        pars.clf_dataset, pars.clf_opt, pars.clf_lr, pars.clf_epochs)

    if not os.path.exists(expdir):
        os.makedirs(expdir)

    print(expdir)
    print(EXP_NAME)

    pars.train_unsupervised = True

    dtype = torch.float32

    data = get_data(pars.datapath, pars.dataset, num_train=500000)
    clf_data = get_data(pars.datapath, pars.clf_dataset, num_train=45000)

    test_acc_all = []
    for rep in range(pars.repeat):
        print("\nRep {}".format(rep+1))

        net, classifier, head = setup_net(pars)
        decoder = setup_decoder(pars)

        net_name = expdir + 'net_rep_{}_'.format(rep+1) + EXP_NAME_NET + '.pt'
        clf_name = expdir + 'clf_rep_{}_'.format(rep+1) + EXP_NAME + '.pt'
        if not os.path.isfile(net_name):
            pars.loadnet = False
        if not os.path.isfile(clf_name):
            pars.loadclf = False
        if pars.loadnet:
            net.load_state_dict(torch.load(net_name))
            print(net_name, "loaded")
        if pars.loadclf:
            classifier.load_state_dict(torch.load(clf_name))
            print(clf_name, "loaded")
        # print(net)
        # print(classifier)
        # print(head)
        # print(decoder)

        val_loss = []
        val_acc = []
        lw_test_acc = []

        head_loss = []

        fix = nn.Sequential()
        model = nn.Sequential(
            net,
            head)
        pars.train_unsupervised = True

        if not pars.loadnet:
            print('Train Net')
            train_model_ae(data, fix, model, decoder, pars, head_loss, criterion_re, criterion_sim, optimizer, vis)

        pars.train_unsupervised = False
        # print(net)
        if not sklearn_classifier:
            # print(classifier)
            if not pars.loadclf:
                print('Train Classifier')
                train_model(clf_data, net, classifier, pars, val_loss, val_acc, clf_criterion, optimizer, vis)
            test_acc = check_accuracy(
                clf_data[4], clf_data[5], net, classifier, pars)
        else:
            print(sklearn_classifier)
            train_model_logistic(clf_data, net, sklearn_classifier, pars)
            test_acc = check_accuracy_logistic(
                clf_data[4], clf_data[5], net, sklearn_classifier, pars)
        print('Rep: %d, te.acc = %.4f' % (rep+1, test_acc))
        lw_test_acc.append(test_acc)

        np.save(expdir+'head_loss_rep_{}_'.format(rep+1) +
                EXP_NAME_NET, head_loss)
        np.save(expdir+'loss_rep_{}_'.format(rep+1) + EXP_NAME, val_loss)
        np.save(expdir+'val.acc_rep_{}_'.format(rep+1) + EXP_NAME, val_acc)
        np.save(expdir+'te.acc_rep_{}_'.format(rep+1) + EXP_NAME, lw_test_acc)

        if pars.epochs and not pars.loadnet:
            torch.save(net.state_dict(), net_name)
        if not sklearn_classifier and not pars.loadclf:
            torch.save(classifier.state_dict(), clf_name)

        test_acc_all.append(lw_test_acc)

    print('\nAll reps test.acc:')
    for acc in test_acc_all:
        print(acc)
    np.save(expdir+'te.acc.all_' + EXP_NAME, test_acc_all)


def proposed_lr(lr0, lr1, i, epochs):
    return lr0*(lr1/lr0)**(i/epochs)


def find_lr_ae(pars, lr0, lr1, n_epochs, criterion_re=None, criterion_sim=None, optimizer=None, vis=None):
    # print(pars)

    pars.train_unsupervised = True

    data = get_data(pars.datapath, pars.dataset, num_train=500000)

    for rep in range(pars.repeat):
        print("\nRep {}".format(rep+1))

        net, classifier, head = setup_net(pars)
        decoder = setup_decoder(pars)
        if pars.loadnet:
            net.load_state_dict(pars.loadnet)

        # print(net)
        # print(classifier)
        # print(head)
        # print(decoder)

        head_loss = []

        fix = nn.Sequential()
        model = nn.Sequential(
            net,
            head)
        pars.train_unsupervised = True

        find_lr_model_ae(data, fix, model, decoder, pars, head_loss, lr0, lr1, n_epochs,  criterion_re, criterion_sim, optimizer, vis)

def find_lr_model_ae(data, fix, model, decoder, pars, ep_loss, lr0, lr1, n_epochs, criterion_re=None, criterion_sim=None, optimizer=None, vis=None):

    device = pars.device
    dtype = torch.float32
    train_dat = data[0]
    train_tar = data[1]

    print(fix)
    print(model)
    print(decoder)

    fix = fix.to(device=device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    decoder = decoder.to(device=device)

    lr = lr0
    if not criterion_re:
        criterion_re = nn.MSELoss()
    if not criterion_sim:
        criterion_sim = TwinMSELoss(pars.batch_size, pars.device)
    params = list(fix.parameters())+list(model.parameters()) + \
        list(decoder.parameters())
    if pars.lam == -1:
        criterion_mtl = MultiTaskLoss(pars.batch_size, pars.device)
        params += list(criterion_mtl.parameters())

    print(criterion_re)
    print(criterion_sim)

    if not optimizer:
        optimizer = torch.optim.Adam(params, lr=lr)
    print(optimizer)

    ep_loss_re = []
    ep_loss_sim = []
    ep_lr = []

    with torch.autograd.set_detect_anomaly(True):
        for e in range(n_epochs):
            running_loss = 0
            running_loss_re = 0
            running_loss_sim = 0
            start_time = time.time()

            for j in np.arange(0, len(train_tar), pars.batch_size):
                lr = proposed_lr(lr0, lr1, e + j / len(train_tar), n_epochs)
                optimizer.param_groups[0]['lr'] = lr

                model.train()  # put model to training mode
                decoder.train()
                if pars.lam == -1:
                    criterion_mtl.train()
                # move to device, e.g. GPU
                x = torch.from_numpy(
                    train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)

                img = x * 0.5 + 0.5

                transform1 = get_BYOL_transforms(True)
                transform2 = get_BYOL_transforms(False)
                x1 = transform1(img)
                x2 = transform2(img)

                x = torch.cat((x1, x2), dim=0)

                with torch.no_grad():
                    x_new = fix(x)
                scores = model(x_new)
                x_re = decoder(scores)

                loss_re = criterion_re(x_re, x_new)
                loss_sim = criterion_sim(scores)

                if pars.lam == -1:  # multi-task loss
                    loss = criterion_mtl(loss_re, loss_sim)
                else:
                    loss = loss_sim+pars.lam*loss_re

                running_loss += loss.item()
                running_loss_re += loss_re.item()
                running_loss_sim += loss_sim.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_time = time.time()-start_time
            running_loss /= (len(train_tar)/pars.batch_size)
            running_loss_re /= (len(train_tar)/pars.batch_size)
            running_loss_sim /= (len(train_tar)/pars.batch_size)
            ep_loss.append(running_loss)
            ep_loss_re.append(running_loss_re)
            ep_loss_sim.append(running_loss_sim)

            ep_lr.append(optimizer.param_groups[0]['lr'])

            print('Epoch %d, loss = %.4f, time: %0.4f' %
                  (e, running_loss, end_time))
            print('reconstruction loss = %.4f, similarity loss: %0.4f' %
                  (running_loss_re, running_loss_sim))
            if pars.lam == -1:
                print(list(criterion_mtl.parameters())[0])
            if vis is not None and e % 5 == 4:
                vis.line(np.array(ep_lr), np.arange(len(ep_lr)),  win="lr", name="lr",
                            opts=dict(title="lr",
                                    xlabel="epochs",
                                    ylabel="lr"))
                vis.line(np.array(ep_loss), np.array(ep_lr),  win="loss_lr", name="loss",
                            opts=dict(title="loss",
                                    xlabel="lr",
                                    ylabel="loss",
                                    xtype="log"))
                # vis.line(np.array(ep_loss_re), np.array(ep_lr),
                #          win="loss_lr", update='append', name="reconstruction loss")
                # vis.line(np.array(ep_loss_sim), np.array(ep_lr),
                #          win="loss_lr", update='append', name="similarity loss")
                vis.line(np.array(ep_loss), np.array(ep_lr),  win="logloss_lr", name="loss",
                            opts=dict(title="loss",
                                    xlabel="lr",
                                    ylabel="loss",
                                    xtype="log",
                                    ytype="log"))
                vis.line(np.array(ep_loss_re), np.array(ep_lr),
                            win="logloss_lr", update='append', name="reconstruction loss")
                vis.line(np.array(ep_loss_sim), np.array(ep_lr),
                            win="logloss_lr", update='append', name="similarity loss")


def find_lr(pars, lr0, lr1, n_epochs, criterion=None, optimizer=None, vis=None):
    # print(pars)

    pars.train_unsupervised = True

    data = get_data(pars.datapath, pars.dataset, num_train=500000)

    for rep in range(pars.repeat):
        print("\nRep {}".format(rep+1))

        net, classifier, head = setup_net(pars)
        if pars.loadnet:
            net.load_state_dict(pars.loadnet)
        # print(net)
        # print(classifier)
        # print(head)

        head_loss = []

        fix = nn.Sequential()
        model = nn.Sequential(
            net,
            head)
        pars.train_unsupervised = True

        find_lr_model(data, fix, model, pars, head_loss, lr0, lr1, n_epochs, criterion, optimizer, vis)



def find_lr_model(data, fix, model, pars, ep_loss, lr0, lr1, n_epochs, criterion=None, optimizer=None, vis=None):

    device = pars.device
    dtype = torch.float32
    train_dat = data[0]
    train_tar = data[1]

    print(fix)
    print(model)

    fix = fix.to(device=device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    lr = lr0
    if not criterion:
        if pars.loss == "BarlowTwins":
            criterion = BarlowTwinsLoss(
                pars.batch_size, pars.lam, pars.device)
        else:
            criterion = SimCLRLoss(pars.batch_size, pars.device)
    params = list(fix.parameters())+list(model.parameters())
    print(criterion)

    if not optimizer:
        optimizer = torch.optim.Adam(params, lr=lr)
    print(optimizer)

    if pars.train_unsupervised:
        n_epochs = pars.epochs
    else:
        n_epochs = pars.clf_epochs

    ep_lr = []

    with torch.autograd.set_detect_anomaly(True):
        for e in range(n_epochs):
            running_loss = 0
            start_time = time.time()

            for j in np.arange(0, len(train_tar), pars.batch_size):
                lr = proposed_lr(lr0, lr1, e + j / len(train_tar), n_epochs)
                optimizer.param_groups[0]['lr'] = lr
                model.train()  # put model to training mode
                # move to device, e.g. GPU
                x = torch.from_numpy(
                    train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)

                img = x * 0.5 + 0.5

                transform1 = get_BYOL_transforms(True)
                transform2 = get_BYOL_transforms(False)
                x1 = transform1(img)
                x2 = transform2(img)

                x = torch.cat((x1, x2), dim=0)


                with torch.no_grad():
                    x_new = fix(x)
                scores = model(x_new)

                loss = criterion(scores)

                running_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            end_time = time.time()-start_time
            running_loss /= (len(train_tar)/pars.batch_size)
            ep_loss.append(running_loss)
            ep_lr.append(optimizer.param_groups[0]['lr'])

            print('Epoch %d, loss = %.4f, time: %0.4f' %
                    (e, running_loss, end_time))

            if vis is not None and e % 5 == 4:
                vis.line(np.array(ep_lr), np.arange(len(ep_lr)),  win="lr", name="lr",
                         opts=dict(title="lr",
                                   xlabel="epochs",
                                   ylabel="lr"))
                vis.line(np.array(ep_loss), np.array(ep_lr),  win="loss_lr", name="loss",
                         opts=dict(title="loss",
                                   xlabel="lr",
                                   ylabel="loss",
                                   xtype="log"))
                vis.line(np.array(ep_loss), np.array(ep_lr),  win="logloss_lr", name="loss",
                         opts=dict(title="loss",
                                   xlabel="lr",
                                   ylabel="loss",
                                   xtype="log",
                                   ytype="log"))
