import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist  # compute distances


class StandardDataset:
    def __init__(self):
        self.protected_attribute_name = ""
        self.privileged_classes = []
        self.fair_variables = []

    def process(
        self,
        train,
        test,
        protected_attribute_name,
        privileged_classes,
        missing_value=[],
        features_to_drop=[],
        categorical_features=[],
        favorable_classes=[],
        normalize=True,
    ):
        cols = [
            x
            for x in train.columns
            if x
            not in (
                features_to_drop
                + [protected_attribute_name]
                + categorical_features
                + ["result"]
            )
        ]

        result = []
        for df in [train, test]:
            # drop nan values
            df = df.replace(missing_value, np.nan)
            df = df.dropna(axis=0)

            # drop useless features
            df = df.drop(columns=features_to_drop)

            # create one-hot encoding of categorical features
            df = pd.get_dummies(df, columns=categorical_features, prefix_sep="=")

            # map protected attributes to privileged or unprivileged
            pos = np.logical_or.reduce(
                np.equal.outer(privileged_classes, df[protected_attribute_name].values)
            )
            df.loc[pos, protected_attribute_name] = 1
            df.loc[~pos, protected_attribute_name] = 0
            df[protected_attribute_name] = df[protected_attribute_name].astype(int)

            # set binary labels
            pos = np.logical_or.reduce(
                np.equal.outer(favorable_classes, df["result"].values)
            )
            df.loc[pos, "result"] = 1
            df.loc[~pos, "result"] = 0
            df["result"] = df["result"].astype(int)

            result.append(df)

        # standardize numeric columns
        for col in cols:
            data = result[0][col].tolist()
            mean = np.mean(data)
            std = np.std(data)
            result[0][col] = (result[0][col] - mean) / std
            result[1][col] = (result[1][col] - mean) / std

        train = result[0]
        test = result[1]
        for col in train.columns:
            if col not in test.columns:
                test[col] = 0
        cols = train.columns
        test = test[cols]
        assert all(
            train.columns[i] == test.columns[i] for i in range(len(train.columns))
        )

        return train, test


    def computeA(self, x, k, lambd=0, print_detail=False, y_hat_values=None):
        n = x.shape[0]
        # x_cpu = x.cpu()
        dist = cdist(x, x, 'euclidean')  # Euclidean distances (needed when device = cuda)
        if lambd != 0:
            max_dist = dist.max()

        if print_detail:
            num_found = 0
            for i in range(n):
                for j in range(n):
                    # print(dist[i][j])
                    # if dist[i][j] < 1.02 and dist[i][j] > 0: # for COMPAS, do not change
                    # if dist[i][j] < 1.02 and dist[i][j] > 1.01: # for adult, do not change
                    if dist[i][j] < 2.5 and dist[i][j] > 0: # for german, do not change
                        # print(dist[i][j])
                        if y_hat_values[i] != y_hat_values[j]:
                            print(i, j, y_hat_values[i], y_hat_values[j], dist[i][j])
                            num_found += 1
            print("num_found:", num_found)

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

        # A = np.zeros([n, n])
        # for it in range(n):
        #     sortedDistIndex = np.ravel(np.argsort(dist[it]))  # np.ravel() compresses dimension
        #     for i in range(0, k):
        #         A[it][sortedDistIndex[i]] = 1
        return A

    def computeYNN(self, x, n, k, lambd=0, print_detail=False, y_hat_values=None):
        # similar_pairs = []
        A = self.computeA(x, k, lambd, print_detail, y_hat_values)
        # print(f"k={k}")
        y_pred_hat = A.dot(y_hat_values)
        y_pred = y_hat_values * k
        temp = np.absolute(y_pred - y_pred_hat)

        # if print_detail:
        #     print("For each data point, how many neighbors is different:")
        #     with np.printoptions(threshold=np.inf):
        #         print(temp)

        temp = temp.sum()
        return (1 - temp * 1.0 / (k * n))


    def analyze(self, df_old, y=None, k=10, log=True, print_detail=False):
        df = df_old.copy()
        if y is not None:
            #df["y hat"] = (y > 0.5).astype(int)
            df["y hat"] = np.where(y > 0.5, 1, 0)

        s = self.protected_attribute_name
        res = dict()
        n = df.shape[0]
        y1 = df.loc[df["result"] == 1].shape[0] / n

        x_idx = df.columns.values.tolist()
        x_idx.remove("result")
        x = df[x_idx].values

        anal_k = 10
        if self.name == "german":
            for k_i in range(2, anal_k + 1, 2):
                for lambd in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20]:
                    # print(f"YNN-k-{k_i}-l-{lambd}")
                    res[f"YNN-k-{k_i}-l-{lambd}"] = self.computeYNN(x, n, k_i, lambd, print_detail, df["y hat"].values)
        else:
            for lambd in [0, 10]:
                res[f"YNN-k-{anal_k}-l-{lambd}"] = self.computeYNN(x, n, anal_k, lambd, print_detail, df["y hat"].values)

        if "y hat" in df.columns:
            yh1s0 = (
                df.loc[(df[s] == 0) & (df["y hat"] == 1)].shape[0]
                / df.loc[df[s] == 0].shape[0]
            )
            yh1s1 = (
                df.loc[(df[s] == 1) & (df["y hat"] == 1)].shape[0]
                / df.loc[df[s] == 1].shape[0]
            )
            yh1y1s0 = (
                df.loc[(df["y hat"] == 1) & (df["result"] == 1) & (df[s] == 0)].shape[0]
                / df.loc[(df["result"] == 1) & (df[s] == 0)].shape[0]
            )
            yh1y1s1 = (
                df.loc[(df["y hat"] == 1) & (df["result"] == 1) & (df[s] == 1)].shape[0]
                / df.loc[(df["result"] == 1) & (df[s] == 1)].shape[0]
            )
            yh0y0s0 = (
                df.loc[(df["y hat"] == 0) & (df["result"] == 0) & (df[s] == 0)].shape[0]
                / df.loc[(df["result"] == 0) & (df[s] == 0)].shape[0]
            )
            yh0y0s1 = (
                df.loc[(df["y hat"] == 0) & (df["result"] == 0) & (df[s] == 1)].shape[0]
                / df.loc[(df["result"] == 0) & (df[s] == 1)].shape[0]
            )

            if print_detail:
                print("yh1s1:", yh1s1)
                print("s1 prop:", (df.loc[df[s] == 1].shape[0] / len(df)))
                print("yh1s0:", yh1s0)
                print("s0 prop:", (df.loc[df[s] == 0].shape[0] / len(df)))

            res["acc"] = df.loc[df["result"] == df["y hat"]].shape[0] / n

            TP = df.loc[(df["result"] == df["y hat"]) & (df["y hat"] == 1)].shape[0]
            TN = df.loc[(df["result"] == df["y hat"]) & (df["y hat"] == 0)].shape[0]
            FP = df.loc[(df["result"] != df["y hat"]) & (df["y hat"] == 1)].shape[0]
            FN = df.loc[(df["result"] != df["y hat"]) & (df["y hat"] == 0)].shape[0]

            res["precision"] = TP / (TP + FP)
            res["recall"] = TP / (TP + FN)
            res["F1"] = 2 / (1 / res["precision"] + 1 / res["recall"])

            res["DP"] = np.abs(yh1s1 - yh1s0)
            tpr = yh1y1s0 - yh1y1s1
            fpr = yh0y0s0 - yh0y0s1
            res["EO"] = np.abs(tpr) * y1 + np.abs(fpr) * (1 - y1)


            # ## code for DCFR
            # fair_variables = self.fair_variables
            # count = (
            #     df.groupby(fair_variables + [s])
            #     .count()["y hat"]
            #     .reset_index()
            #     .rename(columns={"y hat": "count"})
            # )
            # count_y = (
            #     df.groupby(fair_variables + [s])
            #     .sum()["y hat"]
            #     .reset_index()
            #     .rename(columns={"y hat": "count_y"})
            # )
            # count_merge = pd.merge(count, count_y, how="outer", on=fair_variables + [s])
            # count_merge["ratio"] = count_merge["count_y"] / count_merge["count"]
            # count_merge = count_merge.drop(columns=["count", "count_y"])
            # count_merge["ratio"] = (2 * count_merge[s] - 1) * count_merge["ratio"]
            # if len(self.fair_variables) > 0:
            #     result = (
            #         count_merge.groupby(fair_variables)
            #         .sum()["ratio"]
            #         .reset_index(drop=True)
            #         .values
            #     )
            # else:
            #     result = count_merge.sum()["ratio"]
            # ## code for DCFR`

        # ## code for DCFR
        # if len(self.fair_variables) > 0:
        #     fairs = (
        #         df.groupby(self.fair_variables).count()[s].reset_index(drop=True).values
        #     )
        #     fairs = fairs / np.sum(fairs)
        # else:
        #     fairs = 1
        # res["CF"] = np.sum(np.abs(result) * fairs)
        # ## code for DCFR


        if log:
            for key, value in res.items():
                print(key, "=", value)
        return res
