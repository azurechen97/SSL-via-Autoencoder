class PARS:
    def __init__(self, device, datapath, savepath, architecture='CONV', nonlinear='hardtanh', batchsize=500, headsize=64, dataset='Cifar100', loss='SimCLR', optimizer='Adam', lr=0.0001, epochs=800,clf_dataset='Cifar10', clf_loss='CE', clf_opt='Adam', clf_lr=0.0002, clf_epochs=400,repeat=5, loadnet=None, loadclf=None, lam=0.5, auxnonlinear=None):
        self.architecture = architecture  # 'LW', 'CONV'
        self.nonlinear = nonlinear  # 'hartanh','tanh', 'relu'
        self.batch_size = batchsize

        self.headsize = headsize  # head for unsupervised learning

        self.dataset = dataset  # 'Cifar10', 'Cifar100'
        self.loss = loss  # 'SimCLR', 'Hinge'
        self.OPT = optimizer  # 'SGD', 'Adam', Only SGD with RLL
        self.LR = lr
        self.epochs = epochs  # Epochs per layer

        self.clf_dataset = clf_dataset  # 'Cifar10', 'Cifar100'
        self.clf_loss = clf_loss  # 'CE', 'Hinge'
        self.clf_opt = clf_opt
        self.clf_lr = clf_lr
        self.clf_epochs = clf_epochs  # epochs for training classifier

        self.repeat = repeat
        self.device = device
        self.datapath = datapath
        self.savepath = savepath
        self.loadnet = loadnet
        self.loadclf = loadclf

        self.lam = lam
        self.auxnonlinear = auxnonlinear

    def __str__(self):
        res = ""
        for key, val in self.__dict__.items():
            if (key != 'loadnet') and (key != 'loadclf'):
                res += "{}: {}\n".format(key, val)
            else:
                res += "{}: {}\n".format(key, val.keys() if val else val)
        return res
