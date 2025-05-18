from sklearn.preprocessing import normalize
import torch.nn as nn
import numpy as np
import cvxpy as cp
import torch


def cos_sim(fea1, fea2):
    assert fea1.shape[0] == fea2.shape[0]
    fea1 = normalize(fea1)
    fea2 = normalize(fea2)
    similarity = []
    for i in range(fea1.shape[0]):
        similarity.append(np.sqrt(np.sum((fea1[i] - fea2[i]) * (fea1[i] - fea2[i]))))
    return similarity


class convex_hull_cvx_dyn(nn.Module):
    def __init__(self, device):
        super(convex_hull_cvx_dyn, self).__init__()
        self.device = device
        self.mse = torch.nn.MSELoss()

    def forward(self, fea1, fea2, lower=0.0, upper=1.0):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)

        A = nfea2.detach().cpu().numpy()
        XX = torch.tensor(np.zeros((nfea1.shape[0], nfea1.shape[0])), dtype=torch.float32,
                          device=torch.device(self.device))
        for i in range(nfea1.shape[0]):
            y = nfea1[i].detach().cpu().numpy()
            x = cp.Variable(nfea1.shape[0])

            objective = cp.Minimize(cp.sum_squares(x @ A - y))
            constraints = [sum(x) == 1, lower <= x, x <= upper]
            prob = cp.Problem(objective, constraints)
            prob.solve()

            x_tensor = torch.tensor(x.value, dtype=torch.float32, device=torch.device(self.device))
            XX[i] = x_tensor
        distance = -self.mse(torch.mm(XX.detach().to(fea1.device), nfea2), nfea1)
        return distance


class FIM(object):
    def __init__(self, step=10, epsilon=10, alpha=1, random_start=True,
                 loss_type=0, nter=5000, upper=1.0, lower=0.0, device='cpu'):

        self.step = step
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_start = random_start
        self.loss_type = loss_type
        self.lower = lower
        self.upper = upper
        self.nter = nter
        self.LossFunction = convex_hull_cvx_dyn(device)

    def process(self, model, pdata):
        data = pdata.detach().clone()
        original_features = model.forward(data)

        if self.random_start:
            torch.manual_seed(0)
            data_adv = data + torch.zeros_like(data).uniform_(-self.epsilon, self.epsilon)
        else:
            data_adv = data
        data_adv = data_adv.detach()

        for i in range(self.step):
            data_adv.requires_grad_()
            protected_features = model.forward(data_adv)
            dis = cos_sim(protected_features.cpu().detach().numpy(), original_features.cpu().detach().numpy())
            print("[Step %d/%d] Cosine Distance: %s" % (i+1, self.step, [round(v, 4) for v in dis]))

            if i < self.nter:  # init several steps to push adv to the outside of the convexhull
                loss = -self.LossFunction(protected_features, original_features, 1 / pdata.shape[0], 1 / pdata.shape[0])
            else:
                loss = -self.LossFunction(protected_features, original_features, self.lower, self.upper)

            model.zero_grad()
            loss.backward(retain_graph=True)
            grad_step_mean = torch.mean(data_adv.grad, 0, keepdim=True)
            data_adv = data_adv.detach() + self.alpha * grad_step_mean.sign()

            delta = torch.mean(data_adv - data, 0, keepdim=True)

            eta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)
            data_adv = torch.clamp(data + eta, 0, 255).detach()

        return eta
