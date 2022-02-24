import torch

class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, batch_size, lam=0.5, device='cpu'):
        super(BarlowTwinsLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.lam = lam

    def forward(self, output):
        h1, h2 = torch.split(output, self.batch_size)
        # normalize along the batch dimension
        h1 = (h1-h1.mean(0, keepdim=True))/h1.std(0, keepdim=True)
        h2 = (h2-h2.mean(0, keepdim=True))/h2.std(0, keepdim=True)
        N, D = h1.shape

        C = torch.mm(h1.transpose(0, 1), h2)/(N-1)

        mask = torch.eye(D, dtype=torch.bool).to(self.device)
        C_diff = (C - mask.double()).pow(2)
        C_diff[~mask] *= self.lam
        loss = C_diff.sum()

        return loss

class TwinMSELoss(torch.nn.Module):
    def __init__(self, batch_size, device='cpu', reduction = "mean"):
        super(TwinMSELoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.reduction = reduction

    def forward(self, output):
        h1, h2 = torch.split(output, self.batch_size)
        # normalize along the batch dimension
        h1 = (h1-h1.mean(0, keepdim=True))/h1.std(0, keepdim=True)
        h2 = (h2-h2.mean(0, keepdim=True))/h2.std(0, keepdim=True)

        loss = torch.nn.functional.mse_loss(h1,h2, reduction = self.reduction)

        return loss

class MultiTaskLoss(torch.nn.Module):
    def __init__(self, batch_size, device='cpu'):
        super(MultiTaskLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.logsigma = torch.nn.Parameter(torch.zeros(2))

    def forward(self, loss_re, loss_sim):

        loss = 0.5*torch.Tensor([loss_re, loss_sim]) / \
            torch.exp(2*self.logsigma) + self.logsigma
        loss = loss.sum()

        return loss

class SimCLRLoss(torch.nn.Module):
    def __init__(self, batch_size, device='cpu'):
        super(SimCLRLoss, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.mask = self.create_mask(batch_size)
        self.criterion = torch.nn.CrossEntropyLoss()

    # create a mask that enables us to sum over positive pairs only
    def create_mask(self, batch_size):
        mask = torch.eye(batch_size, dtype=torch.bool).to(self.device)
        return mask

    def forward(self, output, tau=0.1):
        norm = torch.nn.functional.normalize(output, dim=1)
        h1, h2 = torch.split(norm, self.batch_size)

        aa = torch.mm(h1, h1.transpose(0, 1))/tau
        aa_s = aa[~self.mask].view(aa.shape[0], -1)
        bb = torch.mm(h2, h2.transpose(0, 1))/tau
        bb_s = bb[~self.mask].view(bb.shape[0], -1)
        ab = torch.mm(h1, h2.transpose(0, 1))/tau
        ba = torch.mm(h2, h1.transpose(0, 1))/tau

        labels = torch.arange(self.batch_size).to(output.device)
        loss_a = self.criterion(torch.cat([ab, aa_s], dim=1), labels)
        loss_b = self.criterion(torch.cat([ba, bb_s], dim=1), labels)

        loss = loss_a+loss_b
        return loss
