import torch
from torch import nn
import math


class FairRepr(nn.Module):
    def __init__(self, model_name, config):
        super(FairRepr, self).__init__()
        dataset = config["dataset"]
        task = config["task"]
        self.fair_coeff = config["fair_coeff"]
        self.fair_coeff_indi = config["fair_coeff_individual"]
        self.task = task
        self.k = config["k"]
        self.gamma = config["gamma"]
        self.name = f"{model_name}_{task}_{dataset}_fair_coeff_{self.fair_coeff}_fair_coeff_indi_{self.fair_coeff_indi}_k_{self.k}"

    def loss_prediction(self, x, y, w):
        return 0

    def loss_audit(self, x, s, f, w):
        return 0

    # # original active version but not use any gamma
    # def loss(self, x, y, s, f, w_pred, w_audit, w_audit_individual):
    #     loss = self.loss_prediction(x, y, w_pred) \
    #            - self.fair_coeff * self.loss_audit(x, s, f, w_audit)\
    #            - self.fair_coeff_indi * self.loss_audit_individual(x, y, w_pred, w_audit_individual, self.k)
    #     return loss
    # # original active version but not use any gamma

    # updated by Hao
    def loss(self, x, y, s, f, w_pred, w_audit, w_audit_individual, verbose=False):
        loss_prd = self.loss_prediction(x, y, w_pred)
        loss_grp = self.loss_audit(x, s, f, w_audit)
        loss_ind = self.loss_audit_individual(x, y, w_pred, w_audit_individual, self.k)
        if self.gamma == 0:
            loss = loss_prd - self.fair_coeff * loss_grp - self.fair_coeff_indi * loss_ind
        else:
            loss = pow(1 - loss_prd, self.gamma) * loss_prd - pow(1 - loss_grp, self.gamma) * loss_grp - pow(1 - loss_ind, self.gamma) * loss_ind

        if verbose:
            print("loss_prd: {}, loss_grp: {}, loss_ind: {}, loss: {}".format(loss_prd, loss_grp, loss_ind, loss))

        # if self.gamma == 0:
        #     loss = self.loss_prediction(x, y, w_pred) \
        #          - self.fair_coeff * self.loss_audit(x, s, f, w_audit) \
        #          - self.fair_coeff_indi * self.loss_audit_individual(x, y, w_pred, w_audit_individual, self.k)
        # else:
        #     loss = pow(1 - self.loss_prediction(x, y, w_pred), self.gamma) * self.loss_prediction(x, y, w_pred) \
        #          - pow(1 - self.loss_audit(x, s, f, w_audit), self.gamma) * self.loss_audit(x, s, f, w_audit) \
        #          - pow(1 - self.loss_audit_individual(x, y, w_pred, w_audit_individual, self.k), self.gamma) * self.loss_audit_individual(x, y, w_pred, w_audit_individual, self.k)
        return loss
    # updated by Hao

    """
    def loss(self, x, y, s, f, w_pred, w_audit, w_audit_individual):
        loss = (self.loss_prediction(x, y, w_pred) \
               - self.fair_coeff * self.loss_audit(x, s, f, w_audit)\
               - self.fair_coeff_indi * self.loss_audit_individual(x, y, w_pred, w_audit_individual, self.k)) / (1+self.fair_coeff +self.fair_coeff_indi )
        return loss
    """
    """
    def loss(self, x, y, s, f, w_pred, w_audit, w_audit_individual):
        loss = max(self.loss_prediction(x, y, w_pred),\
               self.loss_audit(x, s, f, w_audit), \
               self.loss_audit_individual(x, y, w_pred, w_audit_individual, self.k) ).requires_grad_()
        return loss
    """

    def weight_pred(self, df):
        n = df.shape[0]
        res = torch.ones((n, 1)) / n
        return res

    def weight_audit(self, df, s, f):
        return torch.tensor([1.0 / df.shape[0]] * df.shape[0])

    def forward_y(self, x):
        pass

    def forward(self, x):
        self.forward_y(x)
