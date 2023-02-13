import os

import pandas as pd
import requests

from gifair.datasets.standard_dataset import StandardDataset


class GermanDataset(StandardDataset):
    def __init__(self):
        super(GermanDataset, self).__init__()
        self.name = "german"
        self.protected_attribute_name = "A13" # Age
        self.privileged_classes = [1] # aged > 40

        filedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "german")
        # self.download(filedir)
        if not os.path.exists(os.path.join(filedir, "german_train.csv")):
            # if True:
            df = pd.read_csv(os.path.join(filedir, "german.csv"))
            # df = self.preprocess(df)

            categorical_features = ["A1", "A3", "A4", "A6", "A7", "A9", "A10", "A12", "A14", "A15", "A17", "A19", "A20"]
            cols = [
                "A1",
                "A2",
                "A3",
                "A4",
                "A5",
                "A6",
                "A7",
                "A8",
                "A9",
                "A10",
                "A11",
                "A12",
                "A13",
                "A14",
                "A15",
                "A16",
                "A17",
                "A18",
                "A19",
                "A20",
                "A21"
            ]
            df = df[cols].copy()
            df = df.rename(columns={"A21": "result"})
            df.sample(frac=1, random_state=0)
            self.test = df.tail(df.shape[0] // 10 * 3)
            self.train = df.head(df.shape[0] - self.test.shape[0])
            self.train, self.test = super().process(
                self.train,
                self.test,
                categorical_features=categorical_features,
                features_to_drop=[],
                missing_value=["?"],
                favorable_classes=[0],
                protected_attribute_name=self.protected_attribute_name,
                privileged_classes=self.privileged_classes,
            )
            self.train.sample(frac=1, random_state=0)
            n = self.train.shape[0]
            self.val = self.train.tail(n // 10 * 2)
            self.train = self.train.head(n - self.val.shape[0])
            self.train.to_csv(os.path.join(filedir, "german_train.csv"), index=None)
            self.val.to_csv(os.path.join(filedir, "german_val.csv"), index=None)
            self.test.to_csv(os.path.join(filedir, "german_test.csv"), index=None)
        else:
            self.train = pd.read_csv(
                os.path.join(filedir, "german_train.csv"), index_col=False
            )
            self.val = pd.read_csv(
                os.path.join(filedir, "german_val.csv"), index_col=False
            )
            self.test = pd.read_csv(
                os.path.join(filedir, "german_test.csv"), index_col=False
            )

        columns = self.train.columns.values
        self.fair_variables = [ele for ele in columns if "A7" in ele]

def main():
    GermanDataset()


if __name__ == "__main__":
    main()
