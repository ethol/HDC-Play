import numpy as np


def similarity_preserving_expansion(data, data_test, exp_size):
    training_length = len(data)
    comb_data = np.concatenate((data, data_test), axis=0)

    ids = np.random.randint(0, 2, size=(comb_data.shape[1], exp_size), dtype="uint8")

    # vec_0 = np.random.randint(0, 2, size=exp_size, dtype="uint8")
    # vec_1 = np.random.randint(0, 2, size=exp_size, dtype="uint8")
    vec_0 = np.zeros(exp_size, dtype="uint8")
    vec_1 = np.ones(exp_size, dtype="uint8")

    # adjust data
    comb_data = comb_data - np.min(comb_data)
    comb_data = comb_data / np.max(comb_data)

    data_expanded = []
    for d in comb_data:
        new_vec = []

        for i in range(len(d)):
            if d[i] == 0:
                vec = vec_0
            else:
                vec = np.hstack((vec_0[int((d[i]) * exp_size):], vec_1[:int((d[i]) * exp_size)]))
            new_vec.append(np.bitwise_xor(vec, ids[i]))
        new_vec = np.array(new_vec)
        data_expanded.append(new_vec.flatten())

    data_train_expanded = np.array(data_expanded[:training_length])
    data_test_expanded = np.array(data_expanded[training_length:])

    return data_train_expanded, data_test_expanded


def simple_thresholding(data, data_test):
    training_length = len(data)
    comb_data = np.concatenate((data, data_test), axis=0)

    # adjust data
    comb_data = comb_data - np.min(comb_data)
    comb_data = comb_data / np.max(comb_data)

    data_bool = np.round(comb_data).astype("uint8")
    data_bool = data_bool.reshape(data_bool.shape[0], -1)

    #pad for BHV by rounding up to nearest 8
    bhv_length = int(np.ceil(data_bool.shape[1] / 8) * 8)
    to_pad = bhv_length - data_bool.shape[1]
    if to_pad != 0:
        data_bool = np.pad(data_bool, ((0, 0), (0, to_pad)), mode='constant', constant_values=0)
    # print(data_bool.shape, bhv_length)

    data_train_binary = np.array(data_bool[:training_length])
    data_test_binary = np.array(data_bool[training_length:])
    return data_train_binary, data_test_binary
