import random
import sys
from collections import defaultdict
from tools.SideChannelConstants import SideChannelConstants
import numpy as np
from sklearn import preprocessing


def preprocess_data(x_data, method):
    """
    Applies preprocessing to the data based on the specified method.
    :param x_data: Input data to be preprocessed, should be a 2D numpy array
    :param method: Method of preprocessing to apply, can be 'norm', 'scaling', or None
    :return:
    """
    if method == 'norm':     # 'horizontal_standardization':
        print('[LOG] -- Using {} method to preprocessing the data.'.format(method))
        mn = np.repeat(np.mean(x_data, axis=1, keepdims=True), x_data.shape[1], axis=1)
        std = np.repeat(np.std(x_data, axis=1, keepdims=True), x_data.shape[1], axis=1)
        x_data = (x_data - mn)/std
    elif method == 'scaling':    #  'horizontal_scaling':
        print('[LOG] -- Using {} method to preprocessing the data.'.format(method))
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_data.T)
        x_data = scaler.transform(x_data.T).T
    else:
        print('[LOG] -- No preprocessing applied to the data.')
    return x_data


def valueAfterSBox(inp_data_byte, key_byte):
    """
    Return the intermediate value of encryption after the substitution box in the first round of AES encryption
    :param inp_data_byte: Input plaintext byte, should be the same byte position as the key
    :param key_byte: Input key byte, should be the same byte position as the plaintext
    :return: The intermediate byte value of encryption
    """
    inp_data_byte = int(inp_data_byte)
    aesSBox = SideChannelConstants.getAESSBox()
    return aesSBox[inp_data_byte ^ key_byte]


def getOneLabel(text_i, target_byte, key, leakage_model):
    """
    get_one_label returns the appropriate intermediate value given a plaintext, key byte, and leakage model
    :param text_i: Plaintext array input
    :param target_byte: Target byte position (0-15)
    :param key: Key array input
    :param leakage_model: Chosen leakage model to formulate the intermediate value
    :return: Integer representing the intermediate value
    """
    if leakage_model in ['HW', 'ID']:
        label = valueAfterSBox(text_i[target_byte], key[target_byte])
        if leakage_model == 'HW':
            label = SideChannelConstants.hw_aes_sbox[label]
    elif leakage_model == 'HD':
        preSBox = text_i[target_byte] ^ key[target_byte]
        postSBox = valueAfterSBox(text_i[target_byte], key[target_byte])
        label = SideChannelConstants.hammingDistance(preSBox, postSBox)
    elif leakage_model == 'NONE':
        finalLabel = ""
        for i in range(16):
            label = valueAfterSBox(text_i[i], key[i])
            strLabel = bin(label)[2:]
            while len(strLabel) < 8:
                strLabel = "0" + strLabel
            finalLabel += strLabel
        label = [int(bit) for bit in finalLabel]
    return label


def get_labels(plain_text, key, target_byte, leakage_model):
    """
    Returns the labels for the given plaintext, key, target byte, and leakage model for supervised learning.
    :param plain_text: Single plaintext array, 16 bytes
    :param key: Single 16-byte key array
    :param target_byte: Target byte position (0-15) for which the label is to be calculated
    :param leakage_model: Leakage model to use for calculating the label, can be 'HW', 'HD', 'ID', 'NONE'
    :return:
    """
    labels = []
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        label = getOneLabel(text_i, target_byte, key, leakage_model)
        labels.append(label)
    if leakage_model in ['HW', 'HD']:
        labels = np.array(labels)
        try:
            assert(set(labels) == set(list(range(9))))
        except AssertionError:
            print('[LOG] -- Not all class have data: ', set(labels))
    elif leakage_model == 'ID':
        labels = np.array(labels)
        try:
            assert(set(labels) == set(range(256)))
        except AssertionError:
            print('[LOG] -- Not all class have data: ', set(labels))
    elif leakage_model == 'NONE':
        labels = np.array(labels)
    return labels


def shiftData(shiftAmount, attack_window, trace_mat, textin_mat):
    """
    Shifts the data in the trace matrix by a random amount within the specified range to simulate random delays.
    :param shiftAmount: Amount of random delay to apply to the data, should be an integer
    :param attack_window: Attack window before adding random delay, should be a list of two integers [start, end]
    :param trace_mat: Entire trace matrix, should be a 2D numpy array
    :param textin_mat: Entire plaintext matrix, should be a 2D numpy array
    :return: Updated trace matrix after applying the random delay
    """
    start_idx, end_idx = attack_window[0], attack_window[1]
    if shiftAmount is not None and shiftAmount > 0:
        print(f'[LOG] -- Data will be shifted in random range: [0, {shiftAmount}]')
        shifted_traces = []
        for i in range(textin_mat.shape[0]):
            random_int = random.randint(0, shiftAmount)
            trace_i = trace_mat[i, start_idx+random_int:end_idx+random_int]
            shifted_traces.append(trace_i)
        trace_mat = np.array(shifted_traces)
    else:
        print('[LOG] -- No random delay applied to the data')
    return trace_mat[:, start_idx:end_idx]


def unpack_data(whole_pack):
    """
    Unpacks the data from the provided NPZ file object.
    :param whole_pack: NPZ file object (from np.load) containing the data
    :return: traces, plaintext, key arrays
    """
    try:
        traces, plain_text, key = whole_pack['power_trace'], whole_pack['plain_text'], whole_pack['key']
    except KeyError:
        try:
            traces, plain_text, key = whole_pack['power_trace'], whole_pack['plaintext'], whole_pack['key']
        except KeyError:
            traces, plain_text, key = whole_pack['trace_mat'], whole_pack['textin_mat'], whole_pack['key']
    return traces, plain_text, key


def loadDataTraining(whole_pack, attack_window, method, trace_num, shifted=0):
    """
    Loads the data from the provided NPZ, applies preprocessing, and shifts the data if necessary.
    :param whole_pack: Loaded NPZ file object (from np.load) containing the data
    :param attack_window: Window for the attack; string in the format "start_end"
    :param method: Optional preprocessing method to apply, can be 'norm', 'scaling', or None
    :param trace_num: Number of traces to load from the FRONT of the dataset
    :param shifted: Optional integer indicating the amount of random delay to apply to the data
    :return: Loaded and preprocessed traces, plaintext, and key
    """
    if isinstance(attack_window, str):
        tmp = attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]

    traces, plain_text, key = unpack_data(whole_pack)
    traces = traces[:trace_num, :]
    plain_text = plain_text[:trace_num, :]
    traces = shiftData(shifted, attack_window, traces, plain_text)

    if method:
        traces = preprocess_data(traces, method)
    return traces, plain_text, key


def loadDataTesting(whole_pack, attack_window, method, trace_num, shifted=0):
    """
    Loads the data from the provided NPZ, applies preprocessing, and shifts the data if necessary.
    :param whole_pack: Loaded NPZ file object (from np.load) containing the data
    :param attack_window: Window for the attack; string in the format "start_end"
    :param method: Optional preprocessing method to apply, can be 'norm', 'scaling', or None
    :param trace_num: Number of traces to load from the BACK of the dataset
    :param shifted: Optional integer indicating the amount of random delay to apply to the data
    :return: Loaded and preprocessed traces, plaintext, and key
    """
    if isinstance(attack_window, str):
        tmp = attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]

    traces, plain_text, key = unpack_data(whole_pack)
    traces = traces[-trace_num:, :]  # Get from back
    plain_text = plain_text[-trace_num:, :]
    traces = shiftData(shifted, attack_window, traces, plain_text)

    if method:
        traces = preprocess_data(traces, method)
    return traces, plain_text, key


def sanity_check(input_layer_shape, X_profiling):
    if input_layer_shape[1] != X_profiling.shape[1]:
        print("Error: model input shape %d instead of %d is not expected ..." % (
        input_layer_shape[1], len(X_profiling[0])))
        sys.exit(-1)
    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        Reshaped_X_profiling = X_profiling
    elif len(input_layer_shape) == 3:
        # This is a 1D CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
    elif len(input_layer_shape) == 4:
        # This is a 2D CNN: expand the dimensions
        Reshaped_X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1, 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)
    return Reshaped_X_profiling


# Main (unused except for testing)
if __name__ == "__main__":
    pass

