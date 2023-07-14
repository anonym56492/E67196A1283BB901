import torch
import os
from gifair.models.fair_repr import FairRepr
from gifair.utils.loss import weighted_cross_entropy, weighted_mse, weighted_mae, weighted_gifair_grp
from gifair.utils.mlp import MLP
import numpy as np
import math
from scipy.spatial.distance import cdist  # compute distances

import pdb

class LAFTR(FairRepr):
    def __init__(self, config):
        super(LAFTR, self).__init__("LAFTR", config)
        self.encoder = MLP(
            [config["xdim"]] + config["encoder"] + [config["zdim"]], "relu"
        )
        self.prediction = MLP(
            [config["zdim"]] + config["prediction"] + [config["ydim"]],
            "relu",
        )
        self.audit = MLP([config["zdim"]] + config["audit"] + [config["sdim"]], "relu")

        self.audit_individual = MLP([config["zdim"]] + config["audit"] + [config["ydim"]], "relu")

    def forward_y(self, x):
        z = self.forward_z(x)
        y = self.prediction(z)
        y = torch.sigmoid(y)
        return y

    def forward_s(self, x, f):
        z = self.forward_z(x)
        s = self.audit(z)
        s = torch.sigmoid(s)
        return s

    def forward_y_individual(self, x):
        z = self.forward_z(x)
        y = self.audit_individual(z)
        y = torch.sigmoid(y)
        return y

    def forward_z(self, x):
        z = torch.nn.functional.relu(self.encoder(x))
        return z

    def forward(self, x):
        self.forward_y(x)

    def loss_prediction(self, x, y, w):
        y_pred = self.forward_y(x)
        loss = weighted_cross_entropy(w, y, y_pred)
        return loss

    def loss_audit(self, x, s, f, w):
        s_pred = self.forward_s(x, f)
        # loss = weighted_cross_entropy(w, s, s_pred) ## zz's version used this cross-entropy
        loss = weighted_mse(w, s, s_pred) ## original version of LAFTR code used mse
        # loss = weighted_mae(w, s, s_pred) ## however, in the paper of LAFTR, the l1 distance is used, which is weird
        # loss = weighted_gifair_grp(w, s, s_pred)
        return loss

    ## It is found that when A is the weighted matrix (each row normalized to sum = k)
    ## then, the current code exactly works
    def loss_audit_individual(self, x, y, w, A, k):
        y_pred = self.forward_y_individual(x)
        n = y_pred.shape[0]
        # print(torch.mm(A, y_pred).sum(), y_pred.sum())
        temp = torch.absolute(y_pred * k - torch.mm(A, y_pred)) * w[0] / k
        temp = temp.sum().requires_grad_()
        # print(temp, "here")
        return temp


    def weight_audit(self, df_old, s, f):
        df = df_old.copy()
        df["w"] = 0.0
        if self.task == "DP":
            amount = df.loc[df[s] == 1].shape[0]
            df.loc[df[s] == 1, "w"] = 1.0 / amount / 2
            df.loc[df[s] == 0, "w"] = 1.0 / (df.shape[0] - amount) / 2
        # elif self.task == "DPINDIVIDUAL":
        #     amount = df.loc[df[s] == 1].shape[0]
        #     df.loc[df[s] == 1, "w"] = 1.0 / amount / 2
        #     df.loc[df[s] == 0, "w"] = 1.0 / (df.shape[0] - amount) / 2
        elif self.task == "EO":
            for ss in range(2):
                for y in range(2):
                    amount = df.loc[(df[s] == ss) & (df["result"] == y)].shape[0]
                    df.loc[(df[s] == ss) & (df["result"] == y), "w"] = 1.0 / amount / 4
        # elif self.task == "CF":
        #     res = (
        #         df.groupby(f + [s])
        #         .count()["w"]
        #         .reset_index()
        #         .rename(columns={"w": "n_s_f"})
        #     )
        #     df = df.merge(res, on=f + [s], how="left")
        #     df["w"] = 1.0 / df["n_s_f"]

        res = torch.from_numpy(df["w"].values).view(-1, 1)
        res = res / res.sum()
        return res

    def weight_audit_gifair(self, df_old, s, f):
        df = df_old.copy()
        df["w"] = 0.0
        if self.task == "DP":
            amount = df.loc[df[s] == 1].shape[0]
            df.loc[df[s] == 1, "w"] = - 1.0 / amount
            df.loc[df[s] == 0, "w"] = 1.0 / (df.shape[0] - amount)
        elif self.task == "EO":
            for ss in range(2):
                for y in range(2):
                    amount = df.loc[(df[s] == ss) & (df["result"] == y)].shape[0]
                    df.loc[(df[s] == ss) & (df["result"] == y), "w"] = 1.0 / amount / 4

        res = torch.from_numpy(df["w"].values).view(-1, 1)
        return res

    def weight_audit_individual(self, x, k, lambd=0):
        n = x.shape[0]
        x_cpu = x.cpu()
        dist = cdist(x_cpu, x_cpu, 'euclidean')  # Euclidean distances (needed when device = cuda)
        if lambd != 0:
            max_dist = dist.max()

        A = np.zeros([n, n])
        for it in range(n):
            sortedDistIndex = np.ravel(np.argsort(dist[it]))  # np.ravel() compresses dimension
            for i in range(1, k + 1):
                if lambd != 0:
                    rel_dist = dist[it][sortedDistIndex[i]] / max_dist
                    A[it][sortedDistIndex[i]] = pow(1 - rel_dist, lambd)
                else:
                    A[it][sortedDistIndex[i]] = 1
            if lambd != 0:
                it_sum = np.sum(A[it])
                ratio = k / it_sum
                for i in range(1, k + 1):
                    A[it][sortedDistIndex[i]] *= ratio ## normalize each row to make sum to k
        res = torch.from_numpy(A)
        return res

        ### gpu version with torch library, seems much slower than cdist function in scipy
        # dist = torch.cdist(x, x).data.cpu()
        # if lambd != 0:
        #     max_dist = torch.max(dist)

        # dist_sorted = torch.argsort(dist, dim=1)
        # if lambd == 0:
        #     A = torch.zeros(n, n)#.to(device)
        #     for it in range(n):
        #         for i in range(0, k):
        #             A[it][dist_sorted[it][i]] = 1
        #     return A
        # else:
        #     B = torch.zeros(n, k)#.to(device)
        #     for it in range(n):
        #         for i in range(0, k):
        #             B[it][i] = dist[it][dist_sorted[it][i]]
        #     B = torch.pow(torch.add(torch.mul(B, -1 / max_dist), 1), lambd)
        #     B = torch.mul(torch.nn.functional.normalize(B, p=1, dim=1), k)

        #     A = torch.zeros(n, n)#.to(device)
        #     for it in range(n):
        #         for i in range(0, k):
        #             A[it][dist_sorted[it][i]] = B[it][i]
        #     return A

    def predict_only(self):
        self.audit.freeze()
        self.audit_individual.freeze()
        self.prediction.activate()
        self.encoder.activate()

    def audit_only(self):
        self.audit.activate()
        self.audit_individual.freeze()
        self.prediction.freeze()
        self.encoder.freeze()

    def audit_individual_only(self):
        self.audit.freeze()
        self.audit_individual.activate()
        self.prediction.freeze()
        self.encoder.freeze()

    def finetune_only(self):
        self.audit.freeze()
        self.audit_individual.freeze()
        self.prediction.activate()
        self.encoder.freeze()

    def predict_params(self):
        return list(self.prediction.parameters()) + list(self.encoder.parameters())

    def audit_params(self):
        return self.audit.parameters()
    def audit_individual_params(self):
        return self.audit_individual.parameters()
    def finetune_params(self):
        return self.prediction.parameters()
