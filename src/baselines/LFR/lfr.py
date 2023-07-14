
# date finished -> 20.08.2020

import sys
import csv
import os
import json

import pdb

import numpy as np
import pandas as pd
import scipy.optimize as optim
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scipy.spatial.distance import cdist  # compute distances

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.special import softmax

dataset = ''
seed = 0
A_y = 1 ## prediction loss parameter -> fixed to 1
A_z = 1 ## group fairness loss parameter
A_x = 1 ## similarity loss parameter


def get_exec_name(dataset, grp_param, ind_param, seed):
    return f"LFR_{dataset}_grp_{grp_param}_ind_{ind_param}_seed_{seed}"


def loss_x(x_new, x_initial):
    """
    Constrains the mapping to Z to be good description of X.
    Prototpyes should retain as much initial info as possible.

    difference is measured by squared sum of difference


    ARGS:
    x_new - Prototypes
    x_initial - raw data
    """
    return np.mean(np.sum(np.square((x_new - x_initial))))


def loss_y(y_true, y_predicted):
    """
    This loss term requires that the prediction of y is as accurate as possible:

    Computes log loss

    ARGS:
    y_true - (num_examples, )
    y_predicted - (num_examples, )
    """
    # logarithm is undefined in 0 which means y cant be 0 or 1 => we clip it
    y_true = np.clip(y_true, 1e-6, 0.999)
    y_predicted = np.clip(y_predicted, 1e-6, 0.999)

    log_loss = np.sum(y_true * np.log(y_predicted) +
                      (1. - y_true) * np.log(1. - y_predicted)) / len(y_true)

    return -log_loss


def loss_z(M_k_sensitive, M_k_non_sensitive):
    """
    Ensures statistical parity

    Calculates L1 distance

    Args:
    M_k_sensitive - (num_prototypes, )
    M_k_non_sensitive - (num_prototypes, )
    """
    return np.sum(np.abs(M_k_sensitive - M_k_non_sensitive))


def distances(X, v, alpha):
    """
    Calculates distance between initial data and each of the prototypes 
    Formula -> euclidean(x, v * alpha) (alpha is weight for each feature)

    ARGS:
    X - (num_examples, num_features)
    v - (num_prototypes, num_features)
    alpha - (num_features, 1)

    returns:
    dists - (num_examples, num_prototypes)
    """
    num_examples = X.shape[0]
    num_prototypes = v.shape[0]
    dists = np.zeros(shape=(num_examples, num_prototypes))

    # X = X.values  # converting to NumPy, this is needed in case you pass dataframe
    for i in range(num_examples):
        dist = np.square(X[i] - v)  # squarred distance
        dist_alpha = np.multiply(dist, alpha)  # multiplying by weights
        sum_ = np.sum(dist_alpha, axis=1)
        dists[i] = sum_

    return dists


def M_nk(dists):
    """
    define Mn,k as the probability that x maps to v

    Given the definitions of the prototypes as points in
    the input space, a set of prototypes induces a natural
    probabilistic mapping from X to Z via the softmax

    Since we already have distances calcutated we just map them to probabilities

    NOTE:
    minus distance because smaller the distance better the mapping

    ARGS:
    dists - (num_examples, num_prototypes)

    Return :
    mappings - (num_examples, num_prototypes)
    """
    return softmax(-dists, axis=1)  # specifying axis is important


def M_k(M_nk):
    """
    Calculate mean of the mapping for each prototype

    ARGS:
    M_nk - (num_examples, num_prototypes)

    Returns:
    M_k - mean of the mappings (num_prototypes, )
    """
    return np.mean(M_nk, axis=0)


def x_n_hat(M_nk, v):
    """
    Gets new representation of the data, 
    Performs simple dot product

    ARGS:
    M_nk - (num_examples, num_prototypes)
    v - (num_prototypes, num_features)

    Returns:
    x_n_hat - (num_examples, num_features)
    """
    return M_nk @ v


def y_hat(M_nk, w):
    """
    Function calculates labels in the new representation space
    Performs simple dot product

    ARGS:
    M_nk - (num_examples, num_prototypes)
    w - (num_prototypes, )

    returns:
    y_hat - (num_examples, )
    """
    return M_nk @ w


def optim_objective(params, data_sensitive, data_non_sensitive, y_sensitive,
                    y_non_sensitive, inference=False, NUM_PROTOTYPES=10,
                    print_every=100):
    """
    Function gathers all the helper functions to calculate overall loss

    This is further passed to l-bfgs optimizer 

    ARGS:
    params - vector of length (2 * num_features + NUM_PROTOTYPES + NUM_PROTOTYPES * num_features)
    data_sensitive - instances belonging to senstive group (num_sensitive_examples, num_features)
    data_non_sensitive - similar to data_sensitive (num_non_senitive_examplesm num_features)
    y_sensitive - labels for sensitive group (num_sensitive_examples, )
    y_non_sensitive - similar to y_sensitive
    inference - (optional) if True than will return new dataset instead of loss
    NUM_PROTOTYPES - (optional), default 10
    A_x - (optional) hyperparameters for loss_X, default 0.01
    A_y - (optional) hyperparameters for loss_Y, default 1
    A_z - (optional) hyperparameters for loss_Z, default 0.5
    print_every - (optional) how often to print loss, default 100
    returns:
    if inference - False :
    float - A_x * L_x + A_y * L_y + A_z * L_z 
    if inference - True:
    x_hat_sensitive, x_hat_non_sensitive, y_hat_sensitive, y_hat_non_sensitive
    """
    optim_objective.iters += 1

    num_features = data_sensitive.shape[1]
    # extract values for each variable from params vector
    alpha_non_sensitive = params[:num_features]
    alpha_sensitive = params[num_features:2 * num_features]
    w = params[2 * num_features:2 * num_features + NUM_PROTOTYPES]
    v = params[2 * num_features + NUM_PROTOTYPES:].reshape(NUM_PROTOTYPES, num_features)

    dists_sensitive = distances(data_sensitive, v, alpha_sensitive)
    dists_non_sensitive = distances(data_non_sensitive, v, alpha_non_sensitive)

    #print(alpha_non_sensitive)
    #print(v)

    # get probabilities of mappings
    M_nk_sensitive = M_nk(dists_sensitive)
    M_nk_non_sensitive = M_nk(dists_non_sensitive)

    # M_k only used for calcilating loss_y(statistical parity)
    M_k_sensitive = M_k(M_nk_sensitive)
    M_k_non_sensitive = M_k(M_nk_non_sensitive)
    L_z = loss_z(M_k_sensitive, M_k_non_sensitive)  # stat parity

    # get new representation of data
    x_hat_sensitive = x_n_hat(M_nk_sensitive, v)
    x_hat_non_sensitive = x_n_hat(M_nk_non_sensitive, v)
    # calculates how close new representation is to original data
    L_x_sensitive = loss_x(data_sensitive, x_hat_sensitive)
    L_x_non_sensitive = loss_x(data_non_sensitive, x_hat_non_sensitive)

    # get new values for labels
    y_hat_sensitive = y_hat(M_nk_sensitive, w)
    y_hat_non_sensitive = y_hat(M_nk_non_sensitive, w)
    # ensure how good new predictions are(log_loss)
    L_y_sensitive = loss_y(y_sensitive, y_hat_sensitive)
    L_y_non_sensitive = loss_y(y_non_sensitive, y_hat_non_sensitive)

    L_x = L_x_sensitive + L_x_non_sensitive
    L_y = L_y_sensitive + L_y_non_sensitive

    loss = A_x * L_x + A_y * L_y + A_z * L_z
    #print(f'loss on iteration {optim_objective.iters} : {loss}, L_x - {L_x * A_x} L_y - {L_y * A_y} L_z - {L_z * A_z}')

    ### individual accuracy group
    if optim_objective.iters % print_every == 0:
        print(f'loss on iteration {optim_objective.iters} : {loss}, L_x : {L_x * A_x}, L_y : {L_y * A_y}, L_z : {L_z * A_z}')

    if optim_objective.iters == 500:
        if not os.path.exists("saved"):
            os.makedirs("saved")
        dir_dataset = os.path.join("saved", dataset)
        if not os.path.exists(dir_dataset):
            os.makedirs(dir_dataset)
        saved_path = os.path.join(dir_dataset, get_exec_name(dataset, A_z, A_x, seed))
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        name = os.path.join(saved_path, f"params_iter_{optim_objective.iters}")


    # exec_name = get_exec_name(dataset, A_z, A_x, seed)
    # saved_path = os.path.join("saved", exec_name)

    # if not os.path.exists(saved_path):
    #     os.makedirs(saved_path)

    # name = os.path.join(saved_path, f"params_iter_{optim_objective.iters}")


        np.savetxt(name, params)
        print("model saved to", name)

    if not inference:
        return loss
    if inference:
        return x_hat_sensitive, x_hat_non_sensitive, y_hat_sensitive, y_hat_non_sensitive

optim_objective.iters = 0


def get_dp(df):
    n = df.shape[0]
    P_y1_s0 = df[(df[:, -1] == 1) & (df[:, -2] == 0)].shape[0] * 1.0 / n
    P_y1_s1 = df[(df[:, -1] == 1) & (df[:, -2] == 1)].shape[0] * 1.0 / n
    return np.abs(P_y1_s1 - P_y1_s0)


## group (-3), y_pred (y hat) (-2), y_real (result) (-1)
def get_eo(df):
    y1 = df[(df[:, -1] == 1)].shape[0] / df.shape[0]
    P_yh1_y1_s0 = (df[(df[:, -1] == 1) & (df[:, -2] == 1) & (df[:, -3] == 0)].shape[0] * 1.0) / (df[(df[:, -1] == 1) & (df[:, -3] == 0)].shape[0] * 1.0)
    P_yh1_y1_s1 = (df[(df[:, -1] == 1) & (df[:, -2] == 1) & (df[:, -3] == 1)].shape[0] * 1.0) / (df[(df[:, -1] == 1) & (df[:, -3] == 1)].shape[0] * 1.0)
    P_yh0_y0_s0 = (df[(df[:, -1] == 0) & (df[:, -2] == 0) & (df[:, -3] == 0)].shape[0] * 1.0) / (df[(df[:, -1] == 0) & (df[:, -3] == 0)].shape[0] * 1.0)
    P_yh0_y0_s1 = (df[(df[:, -1] == 0) & (df[:, -2] == 0) & (df[:, -3] == 1)].shape[0] * 1.0) / (df[(df[:, -1] == 0) & (df[:, -3] == 1)].shape[0] * 1.0)
    # print(y1, P_yh1_y1_s0, P_yh1_y1_s1, P_yh0_y0_s0, P_yh0_y0_s1)
    return (np.abs(P_yh1_y1_s0 - P_yh1_y1_s1) * y1 + np.abs(P_yh0_y0_s0 - P_yh0_y0_s1) * (1 - y1))

    ''' code from DCFR
    y1 = df.loc[df["result"] == 1].shape[0] / n
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
    tpr = yh1y1s0 - yh1y1s1
    fpr = yh0y0s0 - yh0y0s1
    res["EO"] = np.abs(tpr) * y1 + np.abs(fpr) * (1 - y1)
    '''


def computeA(x, k, lambd=0):
    n = x.shape[0]
    dist = cdist(x, x, 'euclidean')  # Euclidean distances (needed when device = cuda)
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

    return A

def computeYNN(x, n, k, y_hat_values, lambd=0):
    # similar_pairs = []
    A = computeA(x, k, lambd)
    # print(f"k={k}")
    y_pred_hat = A.dot(y_hat_values)
    y_pred = y_hat_values * k
    temp = np.absolute(y_pred - y_pred_hat)
    temp = temp.sum()
    return (1 - temp * 1.0 / (k * n))

## original version not used anymore
def YNN(df, y_pred, k):
    n = df.shape[0]
    k = 10
    dist = cdist(df, df, 'euclidean')  # Euclidean distances
    A = np.zeros([n, n])
    for it in range(n):
        sortedDistIndex = np.ravel(np.argsort(dist[it]))  # np.ravel() compresses dimension
        for i in range(0, k):
            A[it][sortedDistIndex[i]] = 1
    y_pred_hat = A.dot(y_pred)
    y_pred = y_pred * k
    temp = np.absolute(y_pred - y_pred_hat)
    temp = temp.sum()
    return (1 - temp / (k * n))


def main():

    if dataset not in ['compas', 'adult', 'german']:
        print("dataset not supported")
        exit()

    if dataset == 'compas':
        train_file = 'compas_train.csv'
        test_file = 'compas_test.csv'
        sens_attr = 'race'
    elif dataset == 'adult':
        train_file = 'Adult_train_after.csv'
        test_file = 'Adult_test_after.csv'
        sens_attr = 'sex'
    elif dataset == 'german':
        train_file = 'german_train.csv'
        test_file = 'german_test.csv'
        sens_attr = 'A13'

    data = pd.read_csv(os.path.join('datasets', train_file))

    print(f"A_y = {A_y}, A_z = {A_z}, A_x = {A_x}, dataset = {dataset}, seed = {seed}")

    # seperation into sensitive and non sensitive
    data_sensitive = data[(data[sens_attr] == 1)]
    data_non_sensitive = data[(data[sens_attr] == 0)]
    y_sensitive = data_sensitive.result
    y_non_sensitive = data_non_sensitive.result

    #print (f'Dataset contains {data.shape[0]} examples and {data.shape[1]} features')
    #print (f'From which {data_sensitive.shape[0]} belong to sensitive group and {data_non_sensitive.shape[0]} to non nensitive group ')

    del data_sensitive['result']
    del data_sensitive[sens_attr]
    del data_non_sensitive['result']
    del data_non_sensitive[sens_attr]

    # Standard Scaling
    data_sensitive = StandardScaler().fit_transform(data_sensitive)
    data_non_sensitive = StandardScaler().fit_transform(data_non_sensitive)

    NUM_PROTOTYPES = 10
    num_features = data_sensitive.shape[1]

    params = np.random.uniform(size=(num_features * 2 + NUM_PROTOTYPES + NUM_PROTOTYPES * num_features))

    bnd = [] # This is needed for l-bfgs algorithm
    for i, _ in enumerate(params):
        if i < num_features * 2 or i >= num_features * 2 + NUM_PROTOTYPES:
            bnd.append((None, None))
        else:
            bnd.append((0, 1))

    #print(params.shape)
    new_params = optim.fmin_l_bfgs_b(optim_objective, x0=params, epsilon=1e-5,
                                      args=(data_sensitive, data_non_sensitive,
                                            y_sensitive, y_non_sensitive),
                                      bounds=bnd, approx_grad=True, maxfun=5_00,
                                      maxiter=5_00)[0]

    x_hat_senitive, x_hat_nons, y_hat_sens, y_hat_nons = optim_objective(new_params, data_sensitive, data_non_sensitive,
                                            y_sensitive, y_non_sensitive, inference=True)

    """
    X = np.vstack((x_hat_nons,x_hat_senitive))

    Y= np.vstack((y_hat_nons.reshape(-1,1), y_hat_sens.reshape(-1,1) ))
    Y = np.where(Y>0.5,1,0)

    """


    data_test = pd.read_csv(os.path.join('datasets', test_file))

    # seperation into sensitive and non sensitive
    data_sensitive_test = data_test[(data_test[sens_attr] == 1)]
    data_non_sensitive_test = data_test[(data_test[sens_attr] == 0)]
    y_sensitive_test = data_sensitive_test.result
    y_non_sensitive_test = data_non_sensitive_test.result

    Y_real = np.hstack((y_non_sensitive_test, y_sensitive_test))
    Y_real = Y_real.T

    #print (f'Dataset contains {data_test.shape[0]} examples and {data_test.shape[1]} features')
    #print (f'From which {data_sensitive_test.shape[0]} belong to sensitive group and {data_non_sensitive_test.shape[0]} to non nensitive group ')

    del data_sensitive_test['result']
    del data_sensitive_test[sens_attr]
    del data_non_sensitive_test['result']
    del data_non_sensitive_test[sens_attr]

    # Standard Scaling
    data_sensitive_test = StandardScaler().fit_transform(data_sensitive_test)
    data_non_sensitive_test = StandardScaler().fit_transform(data_non_sensitive_test)

    x_hat_senitive, x_hat_nons, y_hat_sens, y_hat_nons = optim_objective(new_params, data_sensitive_test, data_non_sensitive_test,
                                            y_sensitive_test, y_non_sensitive_test, inference=True)

    Y_pred = np.hstack((y_hat_nons, y_hat_sens))
    Y_pred = Y_pred.T
    X_test = np.vstack((data_non_sensitive_test, data_sensitive_test))

    n1 = x_hat_nons.shape[0]
    n2 = x_hat_senitive.shape[0]
    a = np.zeros([n1, 1])
    b = np.ones([n2, 1])
    protect = np.vstack((a, b))

    Y_pred = np.where(Y_pred > 0.5, 1, 0) ## should be this, but why it has very low accuracy?
    # Y_pred = np.where(Y_pred > 0.5, 0, 1)

    acc = accuracy_score(Y_pred.T, Y_real.T)
    # if acc < 0.5:
    #     acc = 1 - acc ## why do this step?
    f1 = f1_score(Y_pred.T, Y_real.T)
    precision = precision_score(Y_pred.T, Y_real.T)
    recall = recall_score(Y_pred.T, Y_real.T)

    # ynn = YNN(X_test, Y_pred.T, 10) ## original version not used
    ynn = computeYNN(X_test, Y_pred.shape[0], 10, Y_pred.T)
    b_ynn = computeYNN(X_test, Y_pred.shape[0], 10, Y_pred.T, lambd=10)

    Y_pred = Y_pred.reshape(-1, 1)
    Y_real = Y_real.reshape(-1, 1)
    dp = get_dp(np.hstack((protect, Y_pred)))
    eo = get_eo(np.hstack((protect, Y_pred, Y_real)))

    print('ACC:', acc)
    print('F1:', f1)
    print('Precision:', precision)
    print('Recall:', recall)
    print('YNN:', ynn)
    print('B-YNN:', b_ynn)
    print('DP:', dp)
    print('EO:', eo)

    res = {"test": {
        "YNN-k-10-l-0": ynn,
        "YNN-k-10-l-10": b_ynn,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "DP": dp,
        "EO": eo
    }}

    config = {
        "fair_coeff": A_z,
        "fair_coeff_individual": A_x,
        "epoch": 500,
        "k": 10,
        "gamma": 0,
        "seed": seed,
        "dataset": dataset,
        "model": "LFR",
        "task": "LFR",
        "lambda": 10.0
    }

    if not os.path.exists("results"):
        os.makedirs("results")
    dir_dataset = os.path.join("results", dataset)
    if not os.path.exists(dir_dataset):
        os.makedirs(dir_dataset)
    result_path = os.path.join(dir_dataset, get_exec_name(dataset, A_z, A_x, seed))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    result_name = os.path.join(result_path, "test_500.json")
    with open(result_name, "w") as f:
        f.write(json.dumps(res, indent=4))
        f.close()

    config_name = os.path.join(result_path, "config.json")
    with open(config_name, "w") as f:
        f.write(json.dumps(config, indent=4))
        f.close()


if __name__ == "__main__":

    dataset = sys.argv[1]
    # A_y = float(sys.argv[1])
    A_z = float(sys.argv[2])
    A_x = float(sys.argv[3])
    seed = int(sys.argv[4])

    np.random.seed(seed) ## original use seed = 509

    main()