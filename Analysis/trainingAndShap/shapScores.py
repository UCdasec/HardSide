# shapScores.py - UC DaSec Logan Reichling - Start 8/19/25
# Calculates shapely values via DeepShap to identify points of interest in traces

import argparse
import os
import sys
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
import shap
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')
tf.compat.v1.disable_v2_behavior()
from tools.SideChannelConstants import LeakageModel
import tools.loadData as loadData
import tools.ASCAD_test_models as ascadTest


def loadDataTrain(inputTracesFile, attackWindow, leakageModel, targetByte, trainTracesNum, ascadV2Type, preprocess=None):
    """
    Load profiling and training data from the given traces file.
    :return: Traces (x values for model, power value sampled),
        labels (y values for model, intermediate AES representation),
        plaintext: original plaintext string,
        key: 128-bit key in hex string,
        inp_shape: Shape of x values traces to compare against model
    """
    datasetFileNameAndExtension = (str(os.path.split(inputTracesFile)[1])).split('.')
    datasetName = datasetFileNameAndExtension[0]
    datasetFileExtension = datasetFileNameAndExtension[1]

    if datasetFileExtension == 'npz':  # Our format
        whole_pack = np.load(inputTracesFile)
        traces, plaintext, key = loadData.loadDataTraining(whole_pack, attackWindow, preprocess, trainTracesNum, shifted=0)
        labels = loadData.get_labels(plaintext, key, targetByte, leakageModel.value)
        if leakageModel != LeakageModel.NONE:
            labels = to_categorical(labels, leakageModel.getEmbeddingSize(leakageModel))
        inp_shape = (traces.shape[1], 1)

    elif datasetFileExtension == 'h5' and datasetName == 'ASCAD':  # ASCAD format:
        print("[NOTICE] -- Loaded ASCADv1 database ignores user-set window, target byte, and preprocess")
        attackWindow = "0_700"
        targetByte = 2
        preprocess = ""
        with h5py.File(inputTracesFile) as in_file:
            traces = np.array(in_file['Profiling_traces/traces'])[:trainTracesNum, :]
            labels = np.array(in_file['Profiling_traces/labels'], dtype='uint8')[:trainTracesNum]
            plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:trainTracesNum]
            key = np.array(in_file['Profiling_traces/metadata'])['key'][0]
            labels = to_categorical(labels, leakageModel.getEmbeddingSize(leakageModel))
            inp_shape = (traces.shape[1], 1)

    elif datasetFileExtension == "h5" and datasetName == 'ASCADv2':
        print("[NOTICE] -- Loaded ASCADv2 database and assuming MultiSCAResNet model...")
        print("[WARNING] -- Ignoring user-set window, target byte, preprocess, trace_num, epochs, and batch size")
        attackWindow = "0_15000"
        targetByte = "1-16"  # Now used for reporting purposes only
        preprocess = ""
        trainTracesNum = 420000
        epochs = 60
        with h5py.File(inputTracesFile) as in_file:
            traces = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)[:trainTracesNum, :]
            tempLabels = np.array(in_file['Profiling_traces/labels'])[:trainTracesNum]
            plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:trainTracesNum]
            key = np.array(in_file['Profiling_traces/metadata'])['key'][0]
            if ascadV2Type == "withPermIDs":
                labels = {'alpha_output': to_categorical(tempLabels['alpha_mask'], num_classes=256),
                          'beta_output': to_categorical(tempLabels['beta_mask'], num_classes=256)}
                for i in range(16):
                    labels['sbox_' + str(i) + '_output'] = to_categorical(tempLabels['sbox_masked'][:, i],
                                                                          num_classes=256)
                for i in range(16):
                    labels['permind_' + str(i) + '_output'] = to_categorical(tempLabels['perm_index'][:, i],
                                                                             num_classes=16)
            elif ascadV2Type == "withoutPermIDs":
                labels = {'alpha_output': to_categorical(tempLabels['alpha_mask'], num_classes=256),
                          'beta_output': to_categorical(tempLabels['beta_mask'], num_classes=256)}
                for i in range(16):
                    labels['sbox_' + str(i) + '_output'] = to_categorical(tempLabels['sbox_masked_with_perm'][:, i],
                                                                          num_classes=256)
            inp_shape = (traces.shape[1], 1)

    elif datasetFileExtension == 'h5' and datasetName == 'CHES':  # CHES format:
        print("[Notice] -- Loaded CHES database for training")
        with h5py.File(inputTracesFile) as in_file:
            # Set attack window:
            tmp = attackWindow.split('_')
            attackWindow = [int(tmp[0]), int(tmp[1])]
            traces = np.array(in_file['Profiling_traces/traces'])[:trainTracesNum, attackWindow[0]:attackWindow[1]]
            plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:trainTracesNum]
            plaintext = plaintext.astype(np.uint8)
            key = np.array(in_file['Profiling_traces/metadata'])['key'][0]  # Just get first key since its fixed
            key = key.astype(np.uint8)
            labels = loadData.get_labels(plaintext, key, targetByte, leakageModel.value)
            labels = to_categorical(labels, leakageModel.getEmbeddingSize(leakageModel))
            inp_shape = (traces.shape[1], 1)
    return traces, labels, plaintext, key, inp_shape


def loadDataTesting(inputTracesFile, attackWindow, leakageModel, targetByte, testTracesNum, ascadV2Type, preprocess=None):
    """
    Load profiling and attack data from the given traces file.
    :return:
    Traces (x values for model, power value sampled),
        labels (y values for model, intermediate AES representation),
        plaintext: original plaintext string,
        key: 128-bit key in hex string,
        inp_shape: Shape of x values traces to compare against model
    """
    datasetFileNameAndExtension = (str(os.path.split(inputTracesFile)[1])).split('.')
    datasetName = datasetFileNameAndExtension[0]
    datasetFileExtension = datasetFileNameAndExtension[1]

    if datasetFileExtension == 'npz':  # Our format
        whole_pack = np.load(inputTracesFile)
        traces, plaintext, key = loadData.loadDataTesting(whole_pack, attackWindow, preprocess, testTracesNum)
        labels = loadData.get_labels(plaintext, key, targetByte, leakageModel.value)
        labels = to_categorical(labels, leakageModel.getEmbeddingSize(leakageModel))
        inp_shape = (traces.shape[1], 1)

    elif datasetFileExtension == 'h5' and datasetName == 'ASCAD':  # ASCADv1 format:
        print("[WARNING] -- Loaded ASCAD database ignores attack window, target byte, preprocess and shifted")
        attackWindow = "0_700"
        targetByte = 2
        preprocess = ""
        shifted = 0
        with h5py.File(inputTracesFile) as in_file:
            traces = np.array(in_file['Attack_traces/traces'])[-testTracesNum:]
            labels = np.array(in_file['Attack_traces/labels'], dtype='uint8')[-testTracesNum:]
            if leakageModel == leakageModel.HW:
                for i, label in enumerate(labels):
                    labels[i] = bin(label).count("1")
            plaintext = np.array(in_file['Attack_traces/metadata'])['plaintext'][-testTracesNum:]
            # Key is same for ASCAD fixed database throughout all rows, in our code we expect a single key array
            key = np.array(in_file['Attack_traces/metadata'])['key'][0]
            labels = to_categorical(labels, leakageModel.getEmbeddingSize(leakageModel))
            inp_shape = (traces.shape[1], 1)

    elif datasetFileExtension == 'h5' and datasetName == 'ASCADv2':
        print("[LOG] -- Loaded ASCADv2 database and assuming MultiSCAResNet model...")
        print("[WARNING] -- Ignoring user-set window, target byte, preprocess, shifted, ")
        attackWindow = "0_15000"
        targetByte = "1-16"  # Now used for reporting purposes only
        preprocess = ""
        shifted = 0
        (X_profiling, Y_profiling), \
            (X_attack, Y_attack), \
            (Metadata_profiling, Metadata_attack) = ascadTest.load_ascad(inputTracesFile, load_metadata=True)
        Y_attack = Y_attack[:testTracesNum]
        if ascadV2Type == "withPermIDs":
            labels = {}
            labels['alpha_output'] = to_categorical(Y_attack['alpha_mask'], num_classes=256)
            labels['beta_output'] = to_categorical(Y_attack['beta_mask'], num_classes=256)
            for i in range(16):
                labels['sbox_' + str(i) + '_output'] = to_categorical(Y_attack['sbox_masked'][:, i], num_classes=256)
            for i in range(16):
                labels['permind_' + str(i) + '_output'] = to_categorical(Y_attack['perm_index'][:, i], num_classes=16)
        elif ascadV2Type == "withoutPermIDs":
            labels = {}
            labels['alpha_output'] = to_categorical(Y_attack['alpha_mask'], num_classes=256)
            labels['beta_output'] = to_categorical(Y_attack['beta_mask'], num_classes=256)
            for i in range(16):
                labels['sbox_' + str(i) + '_output'] = to_categorical(Y_attack['sbox_masked_with_perm'][:, i], num_classes=256)
        traces = X_attack[:testTracesNum, :]
        plaintext = Metadata_attack[:testTracesNum]
        key = Metadata_attack['key'][:testTracesNum]
        inp_shape = (traces.shape[1], 1)

    return traces, labels, plaintext, key, inp_shape


def parseArgs():
    """
    Parse command line arguments if run from command line
    :return: Namespace object containing parsed arguments (i.e. args = parser.parse_args(); args.input;)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_traces', help='Input traces used to train the model')
    parser.add_argument('-m', '--inputModelFile', help='Input TRAINED model for explainability analysis')
    parser.add_argument('-o', '--output_dir', help='Output directory for the results. Blank to use inputModelDir')
    parser.add_argument('-tn', '--train_traces', type=int, help='Number of traces to train the model')
    parser.add_argument('-tt', '--test_traces', type=int, help='Number of traces to test the model')
    parser.add_argument('-tb', '--target_byte', type=int, help='Target byte to attack (0-15). Should be same as during training')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'HD', 'ID'}, help='Leakage model of the network')
    parser.add_argument('-aw', '--attack_window', help='Attack window (POI window) for the traces')
    parser.add_argument("-mt", "--model_type", choices={"CNN", "MLP", "ResNetSingle", "ASCADv2"}, help="Model type for training.")
    parser.add_argument('-ascadv2Type', '--ascadv2_type', choices={'withPermIDs', 'withoutPermIDs'},
                        help="For use with ASCADv2 only. Selects the type of MultiSCAResNet.")
    parser.add_argument('-padTo', '--padTo', type=int, default=None,
                        help='Pad the attack window up to this width with zeroes')
    parser.add_argument('-v', '--verbose', action='store_true', help='Include for verbose output')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scaling'},
                        help='Preprocessing method (unused)')
    opts = parser.parse_args()
    return opts


# Main
if __name__ == "__main__":
    startTestTimer = time.time()
    cmdArgs = parseArgs()
    required_args = ['train_traces', 'test_traces', 'target_byte', 'attack_window', 'model_type']
    for req in required_args:
        if getattr(cmdArgs, req) is None:
            print(f"Error: Required argument --{req} is missing")
            sys.exit(1)
    if not os.path.isfile(cmdArgs.input_traces):
        print(f"Error: Input traces file '{cmdArgs.input_traces}' does not exist")
        sys.exit(1)
    if not os.path.isfile(cmdArgs.inputModelFile):
        print(f"Error: Model file '{cmdArgs.inputModelFile}' does not exist")
        sys.exit(1)
    outputDir = None
    if cmdArgs.output_dir:
        if not os.path.exists(cmdArgs.output_dir):
            os.makedirs(cmdArgs.output_dir)
            outputDir = cmdArgs.output_dir
    else:
        outputDir = cmdArgs.inputModelFile

    if cmdArgs.leakage_model == 'HW':
        leakageModel = LeakageModel.HW
    elif cmdArgs.leakage_model == 'ID':
        leakageModel = LeakageModel.ID
    elif cmdArgs.leakage_model == 'HD':
        leakageModel = LeakageModel.HD

    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    X_profiling, Y_profiling, _, _, _ = loadDataTrain(cmdArgs.input_traces, cmdArgs.attack_window,
                                                                     leakageModel, cmdArgs.target_byte,
                                                                     cmdArgs.train_traces, cmdArgs.ascadv2_type)

    X_attack, Y_attack, _, _, _ = loadDataTesting(cmdArgs.input_traces, cmdArgs.attack_window,
                                                                     leakageModel, cmdArgs.target_byte,
                                                                     cmdArgs.test_traces, cmdArgs.ascadv2_type)

    if cmdArgs.padTo is not None:
        X_profiling = np.pad(X_profiling, [(0, 0), (0, cmdArgs.padTo - X_profiling.shape[1])], constant_values=0)
        X_attack = np.pad(X_attack, [(0, 0), (0, cmdArgs.padTo - X_attack.shape[1])], constant_values=0)
        inp_shape = (X_attack.shape[1], 1)

    attack_window = None
    if isinstance(cmdArgs.attack_window, str):
        tmp = cmdArgs.attack_window.split('_')
        attack_window = [int(tmp[0]), int(tmp[1])]
    window_size = attack_window[1] - attack_window[0]
    X_profiling = np.reshape(X_profiling, (cmdArgs.train_traces, X_profiling.shape[1], 1))
    X_attack = np.reshape(X_attack, (cmdArgs.test_traces, X_attack.shape[1], 1))

    # Load the model
    model = load_model(cmdArgs.inputModelFile, compile=False)

    restoreShapValues = False
    if not restoreShapValues:
        # Create SHAP explainer and compute values
        # if cmdArgs.model_type in ["ResNetSingle", "ASCADv2"]:
        #     shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
        explainer = shap.GradientExplainer(model, X_profiling)
        shap_values = explainer.shap_values(X_attack)

        # Convert to numpy array and compute mean absolute value per time step
        shap_array = np.array(shap_values)  # Shape: (n_classes, n_samples, window_size, 1)
        mean_abs_shap = np.mean(np.abs(shap_array), axis=(0, 1, 3))  # Shape: (window_size,)
        np.savez_compressed("shap_restore.npz", y=mean_abs_shap)
    else:
        mean_abs_shap = np.load("shap_restore.npz")['y']

    # Create the plot
    plt.figure(figsize=(4, 3))
    plt.plot(range(window_size), mean_abs_shap[:window_size])
    plt.xlabel('Timestamp')
    plt.ylabel('Mean |SHAP Value|')
    # plt.title('Feature Importance Over Attack Window')
    # plt.grid(True)
    plt.savefig('shap_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Make cool plot now with shap bars and black trace line

    # plt.plot(np.arange(attack_window[0], attack_window[1]), np.average(X_profiling, axis=0))
    # plt.figure(figsize=(8,6))
    # plt.savefig(os.path.join(os.path.split(outputDir)[0], "test.png"), format='png')

    endTestTimer = time.time()
    print(endTestTimer-startTestTimer)