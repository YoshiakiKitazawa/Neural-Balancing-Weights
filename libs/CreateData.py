import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
ohenc_W5 = OneHotEncoder(sparse=False, drop='first')


def gen(train_data_size: int,
        test_data_size: int,
        experiment_type: str):
    sample_size = train_data_size + test_data_size
    denominator_Y = 50
    myu_vec = [-0.5, 1, 0, 1]
    sigma_mat = np.repeat(0.0, 4*4).reshape(4, 4)
    np.fill_diagonal(sigma_mat, 1)
    W_mat = np.random.multivariate_normal(myu_vec, sigma_mat, sample_size)
    W1 = W_mat[:, 0]
    W2 = W_mat[:, 1]
    W3 = W_mat[:, 2]
    W4 = W_mat[:, 3]
    W5 = np.random.choice(a=3, size=sample_size, p=(0.7, 0.15, 0.15))
    W5_mat = W5.reshape(sample_size, 1)
    ohenc_W5.fit(W5_mat)
    myuA = (5*np.abs(W1) + 6*np.abs(W2) + np.abs(W4)
            + 1*np.abs(W5 == 1) + 5*np.abs(W5 == 2))
    A = np.random.noncentral_chisquare(3, myuA)
    C = (1 + 3.5**2) + (1 + 24**2)
    Y = (((-0.15*A**2 + A*(W1**2+W2**2) - 15
          + (W1 + 3)**2 + 2*(W2 - 25)**2 + W3)
          - C)/denominator_Y + np.random.normal(0, 1, sample_size))
    X1_1 = np.exp(W1/2)
    X1_2 = W2/(1 + np.exp(W1)) + 10
    X1_3 = (W1*W3)/25 + 0.6
    X4 = (W4 - 1)**2
    X5 = ohenc_W5.transform(W5_mat)

    X1_1 = np.reshape(X1_1.astype(np.float32),
                      (sample_size, 1))
    X1_2 = np.reshape(X1_2.astype(np.float32),
                      (sample_size, 1))
    X1_3 = np.reshape(X1_3.astype(np.float32),
                      (sample_size, 1))
    X4 = np.reshape(X4.astype(np.float32),
                    (sample_size, 1))
    X5 = np.reshape(X5.astype(np.float32),
                    (sample_size, X5.shape[1]))
    A = np.reshape(A.astype(np.float32),
                   (sample_size, 1))
    Y = np.reshape(Y.astype(np.float32),
                   (sample_size, 1))
    original_explanatories = [A, X1_1, X1_2, X1_3, X4, X5]
    original_response = Y

    orig_explanatories_train = []
    orig_explanatories_test = []
    for _idx in range(len(original_explanatories)):
        i_d = original_explanatories[_idx]
        i_train, i_test = train_test_split(
            i_d,
            test_size=test_data_size,
            train_size=train_data_size,
            random_state=0)
        orig_explanatories_train.append(i_train)
        orig_explanatories_test.append(i_test)

    response_train, response_test = train_test_split(
          original_response,
          test_size=test_data_size,
          train_size=train_data_size,
          random_state=0)

    A_train = orig_explanatories_train[0]
    X1_to_3_train_list = orig_explanatories_train[1:]
    X1_to_3_train = np.concatenate(X1_to_3_train_list, axis=1)

    A_test = orig_explanatories_test[0]
    X1_to_3_test_list = orig_explanatories_test[1:]
    X1_to_3_test = np.concatenate(X1_to_3_test_list, axis=1)

    # for explanatories of train data
    explanatories_train = ([A_train] + [X1_to_3_train])
    explanatories_test = ([A_test] + [X1_to_3_test])

    # for explanatories of test data
    # (generated from shuffling by the index)
    if experiment_type == 'Experiment1':
        # shuffle each of A and X1_to_3_train by the index, respectively
        r_index_A = np.random.choice(
            test_data_size, size=test_data_size, replace=False)
        A_intervention = A_test[r_index_A, :]
        r_index_X_all_intervention = np.random.choice(
            test_data_size, size=test_data_size, replace=False)
        X1_to_3_intervention = X1_to_3_test[r_index_X_all_intervention, :]
        X1_1_intervention = X1_to_3_test[:, [0]]
        X1_2_intervention = X1_to_3_test[:, [1]]
        X1_3_intervention = X1_to_3_test[:, [2]]
        X2_intervention = X1_to_3_test[:, [3]]
        X3_intervention = X1_to_3_test[:, 4:]
        explanatories_intervention = [
             A_intervention,
             X1_1_intervention,
             X1_2_intervention,
             X1_3_intervention,
             X2_intervention,
             X3_intervention]
    elif experiment_type == 'Experiment2':
        # shuffle each of A, X1, X2 and X3 by the index, respectively
        r_index_A = np.random.choice(
            test_data_size, size=test_data_size, replace=False)
        A_intervention = A_test[r_index_A, :]
        r_index_X1 = np.random.choice(
            test_data_size, size=test_data_size, replace=False)
        X1_1_intervention = X1_to_3_test[r_index_X1, :][:, [0]]
        X1_2_intervention = X1_to_3_test[r_index_X1, :][:, [1]]
        X1_3_intervention = X1_to_3_test[r_index_X1, :][:, [2]]
        r_index_X2 = np.random.choice(
            test_data_size, size=test_data_size, replace=False)
        X2_intervention = X1_to_3_test[r_index_X2, :][:, [3]]
        r_index_X3 = np.random.choice(
            test_data_size, size=test_data_size, replace=False)
        X3_intervention = X1_to_3_test[r_index_X3, 4:]
        explanatories_intervention = [
             A_intervention,
             X1_1_intervention,
             X1_2_intervention,
             X1_3_intervention,
             X2_intervention,
             X3_intervention]

    # inverse transformation for calcurating true response value Y
    W1_intervention = 2*np.log(X1_1_intervention.flatten())
    W2_intervention = (X1_2_intervention.flatten() - 10)*(
      1+X1_1_intervention.flatten()**2)
    W3_intervention = (
      25*(X1_3_intervention.flatten() - 0.6)
      / X1_1_intervention.flatten())

    # calcurate true response value Y
    true_response_intervention = (
      - 0.15*A_intervention.flatten()**2
      + A_intervention.flatten()*(W1_intervention**2 + W2_intervention**2)
      - 15
      + (W1_intervention + 3)**2
      + 2*(W2_intervention - 25)**2
      + W3_intervention - C)/denominator_Y

    return [explanatories_train, response_train,
            explanatories_test, response_test,
            explanatories_intervention,
            true_response_intervention]
