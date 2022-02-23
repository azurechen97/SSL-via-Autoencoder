import torch.nn as nn

def setup_net(pars):
    net = nn.Sequential()
    classifier = nn.Sequential()
    head = nn.Sequential()

    if pars.clf_dataset == 'Cifar100':
        NUM_CLASS = 100
    elif pars.clf_dataset == 'Cifar10':
        NUM_CLASS = 10
    else:
        NUM_CLASS = 1000

    HW = 32
    NUM_CHANNEL = 32
    pars.NUM_LAYER = 5

    if pars.nonlinear == 'hardtanh':
        nonlinear = nn.Hardtanh()
    else:
        nonlinear = nn.ReLU()

    for i in range(pars.NUM_LAYER):
        layer = nn.Sequential()

        if i==0:
            layer.add_module('conv', nn.Conv2d(3,int(NUM_CHANNEL),3,padding=1))
            layer.add_module('activation', nonlinear)
        
        elif (i == 1) or (i == 3):
            layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL),3,padding=1))
            layer.add_module('maxpool', nn.MaxPool2d(2))
            HW /= 2

        elif i == 2:
            layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*2),3,padding=1))
            layer.add_module('activation', nonlinear)
            NUM_CHANNEL *= 2

        else:
            layer.add_module('conv', nn.Conv2d(int(NUM_CHANNEL),int(NUM_CHANNEL*8),3,padding=1))
            layer.add_module('maxpool', nn.MaxPool2d(2))
            NUM_CHANNEL *= 8
            HW /= 2
    
        net.add_module('layer%d'%i, layer)

    aux = nn.Sequential(
        nn.Flatten(),
    )
    aux.add_module('fc', nn.Linear(int(NUM_CHANNEL*HW*HW), NUM_CLASS))

    auxhead = nn.Sequential(
        nn.Flatten(),
    )
    auxhead.add_module('fc', nn.Linear(
        int(NUM_CHANNEL*HW*HW), pars.headsize))
    if pars.auxnonlinear == 'tanh':
        auxhead.add_module('activation', nn.Tanh())

    classifier.add_module('aux%d'%i, aux)

    head.add_module('auxhead%d'%i, auxhead)


    return net, classifier, head

