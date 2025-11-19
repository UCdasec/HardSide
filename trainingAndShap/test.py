# test.py - Refactor start 12/16/2023 - UC DASEC - Logan Reichling

import argparse
import hashlib
import os
import re
import sys
import time
from datetime import date
import h5py
import numpy as np
import tools.ASCAD_test_models as ascadTest
import tools.key_rank_new as key_rank
import tools.loadData as loadData
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.python.platform import build_info
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel('ERROR')
from tools.SideChannelConstants import SideChannelConstants, LeakageModel, ModelType


class SideChannelTester:
    """
    Class to test a saved model against one of the testing databases Attack traces
    """
    def __init__(self, inputTracesFile: str, inputModelDir: str, outputDir: str,
                 targetByte: int, leakageModel: str, attackWindow: str, testTraceNum: int,
                 preprocess: str = "", shifted: int = 0, chesAttackOption: str = None, ascadV2Type: str = None,
                 verbose: bool = False, plaintextAttack: bool = False, bulk: bool = False, padTo: int = None):
        """
        Initialize a SideChannelTester object by which to test SCA models against attack traces
        :param inputTracesFile: Direct filepath to the .npz file containing the attack traces
        :param inputModelDir:   Directory filepath to the top level model directory to be tested
        :param outputDir:       Directory filepath to the top level output directory for test results (usually same)
        :param targetByte:      Byte number of the key to be tested (0-15)
        :param leakageModel:    String representing the leakage model to be used ('HW' or 'ID')
        :param attackWindow:    String representing the attack window to be used (e.g. '1200_2200')
        :param testTraceNum:    Number of traces to test against (e.g. 10000)
        :param preprocess:      String representing the preprocessing to be used ('', 'norm', or 'scaling')
        :param shifted:         Integer representing the number of traces to shift the attack window by
        :param chesAttackOption String required for CHES dataset testing, specifies same or cross-device setting
        :param ascadV2Type      For use with ASCADv2 dataset only. Selects type of MultiSCAResNet model to utilize.
        :param verbose:         Boolean representing whether to print optional, verbose output
        :param bulk:            Bulk mode option
        :param padTo:           Widen traces using padded zeroes. Needed for some models
        """
        self.inputTracesFile = inputTracesFile
        self.inputModelDir = inputModelDir
        self.outputDir = outputDir
        if self.outputDir is None or self.outputDir:
            print('[LOG] -- Output test results will be located in the Input Model Directory.')
            self.outputDir = self.inputModelDir

        self.targetByte = targetByte
        self.attackWindow = attackWindow
        self.testTracesNum = testTraceNum
        self.preprocess = preprocess
        self.shifted = shifted
        self.chesAttackOption = chesAttackOption
        self.ascadV2Type = ascadV2Type
        self.verbose = verbose
        self.plaintextAttack = plaintextAttack
        self.bulk = bulk
        self.padTo = padTo

        # Calculate additional parameters
        if leakageModel == 'HW':
            self.leakageModel = LeakageModel.HW
        elif leakageModel == 'ID':
            self.leakageModel = LeakageModel.ID
        elif leakageModel == 'HD':
            self.leakageModel = LeakageModel.HD

        # Extract testing dataset name for later metadata use (e.g. X1_K1_200k)
        _, fileName = os.path.split(self.inputTracesFile)
        self.datasetName = fileName.split('.')[0]
        self.datasetFileExtension = fileName.split('.')[1]

        # Store exact model file path and extension, then load model
        if not self.bulk:
            exitFlag = False
            if not os.path.exists(self.inputTracesFile):
                print('[FATAL] -- Input traces file does not exist.')
                exitFlag = True
            if not os.path.exists(self.inputModelDir):
                print('[FATAL] -- Input model directory does not exist.')
                exitFlag = True
            if not (os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.tflite')) or
                    os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.h5'))):
                print(f'[FATAL] -- Input model directory {self.inputModelDir} does not contain a model file.')
                exitFlag = True
            if os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.tflite')):
                self.inputModelFile = os.path.join(self.inputModelDir, 'model', 'best_model.tflite')
                self.modelType = ModelType.TFLITE
                self.model = tf.lite.Interpreter(model_path=self.inputModelFile)
                self.model.allocate_tensors()
            elif os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.h5')):
                self.inputModelFile = os.path.join(self.inputModelDir, 'model', 'best_model.h5')
                self.modelType = ModelType.H5
                try:
                    self.model = load_model(self.inputModelFile)
                except ValueError:
                    self.model = load_model(self.inputModelFile, compile=False)
                    self.model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
            else:
                print('[FATAL] -- Input model file does not exist.')
                exit(1)
            if self.inputModelFile is None or not os.path.exists(self.inputModelFile):
                print('[FATAL] -- Input model file does not exist in expected format.')
                exitFlag = True
            if self.modelType is None or self.modelType not in ModelType:
                print('[FATAL] -- Input model file must be either .h5 or .tflite.')
                exitFlag = True
            if self.model is None:
                print('[FATAL] -- Input model file could not be loaded.')
                exitFlag = True
            if exitFlag:
                exit(1)

        else:  # Bulk mode is currently used to test many models in a directory trained on the same dataset
            # Add the file path of every h5 file in the directory to a list
            self.h5Files = [os.path.join(self.inputModelDir, f) for f in os.listdir(self.inputModelDir) if f.endswith('.h5')]
            self.tfliteFiles = [os.path.join(self.inputModelDir, f) for f in os.listdir(self.inputModelDir) if f.endswith('.tflite')]
            if len(self.h5Files) == 0 and len(self.tfliteFiles) == 0:
                print('[FATAL] -- Input model files do not exist at input directory.')
                exit(1)
        # Ensure that the other required parameters are valid
        self.verifyParameters()

    def verifyParameters(self):
        """
        Verify that the testing parameters are valid. EXIT if invalid.
        :return: None
        """
        exitFlag = False
        if self.inputTracesFile is None or not os.path.exists(self.inputTracesFile):
            print('[FATAL] -- Input traces file does not exist.')
            exitFlag = True
        if self.inputModelDir is None or not os.path.exists(self.inputModelDir):
            print('[FATAL] -- Input model directory does not exist.')
            exitFlag = True
        if self.outputDir is not None and not os.path.exists(self.outputDir):
            print('[INFO] -- Output directory not specified, using input model directory...')
        if self.datasetFileExtension != 'npz' and self.datasetFileExtension != 'h5':
            print(f'[FATAL] -- Input traces file {self.datasetName} must be a .npz or .h5 file.')
            exitFlag = True
        datasetMatch = re.compile(SideChannelConstants.DATASET_NAMING_CONVENTION)
        if (datasetMatch.match(self.datasetName) is None
                and self.datasetName != 'ASCAD' and self.datasetName != 'CHES' and self.datasetName != 'ASCADv2' and
                self.datasetName[:9] != 'Synthetic' and self.datasetName[:6] != 'SMAesH'):
            print(f'[WARN] -- Input traces "{self.datasetName}" file unrecognized. Modify python file if needed.')
            # exitFlag = True
        if self.datasetName == "CHES" and self.chesAttackOption is None:
            print(f'[FATAL] -- ChesAttackOption option unfilled for the CHES database, use "-chesAttack same" or cross')
            exitFlag = True
        if self.datasetName != "ASCADv2":
            if int(self.targetByte) < 0 or (self.targetByte > 15):
                print(f'[FATAL] -- Target byte {self.targetByte} must be between 0 and 15.')
                exitFlag = True
            if self.ascadV2Type is not None:
                print(f"[WARNING] -- AscadV2Type variable set without ASCADv2 dataset. Option will be ignored.")
        elif self.datasetName == "ASCADv2":
            if self.ascadV2Type is None:
                print(f"[FATAL] -- AscadV2Type variable required with ASCADv2 dataset.")
                exit(1)
            if self.ascadV2Type not in ['withPermIDs', 'withoutPermIDs']:
                print(f"[FATAL] -- AscadV2Type not 'withPermIDs' or 'withoutPermIDs'.")
                exitFlag = True
        if self.leakageModel is None or self.leakageModel not in LeakageModel:
            print('[FATAL] -- Leakage model must be either "HW", "HD", or "ID".')
            exitFlag = True
        if self.testTracesNum < 0:
            print('[FATAL] -- Test trace number must be greater than 0.')
            exitFlag = True
        if self.preprocess not in ['', 'norm', 'scaling']:
            print('[FATAL] -- Preprocess must be either norm or scaling.')
            exitFlag = True
        if exitFlag:
            exit(1)  # Exit if any of the above parameters are invalid


    def load_data(self):
        """
        Class function designed for use with an initialized SideChannelTester object.
        Load profiling and attack data from the given npz traces or h5 ASCAD traces file.
        :return:
        Traces (x values for model, power value sampled),
            labels (y values for model, intermediate AES representation),
            plaintext: original plaintext string,
            key: 128-bit key in hex string,
            inp_shape: Shape of x values traces to compare against model
        """
        if self.datasetFileExtension == 'npz':  # Our format
            whole_pack = np.load(self.inputTracesFile)
            traces, plaintext, key = loadData.loadDataTesting(whole_pack, self.attackWindow, self.preprocess,
                                                              self.testTracesNum, self.shifted)
            labels = loadData.get_labels(plaintext, key, self.targetByte, self.leakageModel.value)
            labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
            inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == 'h5' and self.datasetName == 'ASCAD':  # ASCADv1 format:
            print("[WARNING] -- Loaded ASCAD database ignores attack window, target byte, preprocess and shifted")
            self.attackWindow = "0_700"
            self.targetByte = 2
            self.preprocess = ""
            self.shifted = 0
            with h5py.File(self.inputTracesFile) as in_file:
                traces = np.array(in_file['Attack_traces/traces'])[-self.testTracesNum:]
                labels = np.array(in_file['Attack_traces/labels'], dtype='uint8')[-self.testTracesNum:]
                if self.leakageModel == self.leakageModel.HW:
                    for i, label in enumerate(labels):
                        labels[i] = bin(label).count("1")
                plaintext = np.array(in_file['Attack_traces/metadata'])['plaintext'][-self.testTracesNum:]
                # Key is same for ASCAD fixed database throughout all rows, in our code we expect a single key array
                key = np.array(in_file['Attack_traces/metadata'])['key'][0]
                labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
                inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == 'h5' and self.datasetName == 'CHES':  # CHES format:
            print("[Notice] -- Loaded CHES database for testing. See notes in code!")
            with h5py.File(self.inputTracesFile) as in_file:
                # Set attack window:
                tmp = self.attackWindow.split('_')
                attackWindow = [int(tmp[0]), int(tmp[1])]
                if self.chesAttackOption == 'same':  # Same-device traces are within same branch of h5 file, pull 5000 from the back
                    # This assumes that only 40000 were used for training
                    print("Utilizing same-device traces for testing CHES database")
                    traces = np.array(in_file['Profiling_traces/traces'])[-self.testTracesNum:,
                             attackWindow[0]:attackWindow[1]]
                    plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][-self.testTracesNum:]
                    plaintext = plaintext.astype(np.uint8)
                    key = np.array(in_file['Profiling_traces/metadata'])['key'][0]  # Just get first key since its fixed
                    key = key.astype(np.uint8)
                    labels = loadData.get_labels(plaintext, key, self.targetByte, self.leakageModel.value)
                    labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
                    inp_shape = (traces.shape[1], 1)
                elif self.chesAttackOption == 'cross':  # Cross-device traces for testing are in attack branch h5 file
                    # There are 5000 traces in this branch
                    print("Utilizing cross-device traces for testing CHES database")
                    traces = np.array(in_file['Attack_traces/traces'])[:self.testTracesNum,
                             attackWindow[0]:attackWindow[1]]
                    plaintext = np.array(in_file['Attack_traces/metadata'])['plaintext'][:self.testTracesNum]
                    plaintext = plaintext.astype(np.uint8)
                    key = np.array(in_file['Attack_traces/metadata'])['key'][0]  # Just get first key since its fixed
                    key = key.astype(np.uint8)
                    labels = loadData.get_labels(plaintext, key, self.targetByte, self.leakageModel.value)
                    labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
                    inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == 'h5' and self.datasetName == 'ASCADv2':
            print("[LOG] -- Loaded ASCADv2 database and assuming MultiSCAResNet model...")
            print("[WARNING] -- Ignoring user-set window, target byte, preprocess, shifted, ")
            self.attackWindow = "0_15000"
            self.targetByte = "1-16"  # Now used for reporting purposes only
            self.preprocess = ""
            self.shifted = 0
            (X_profiling, Y_profiling), \
                (X_attack, Y_attack), \
                (Metadata_profiling, Metadata_attack) = ascadTest.load_ascad(self.inputTracesFile, load_metadata=True)
            Y_attack = Y_attack[:self.testTracesNum]
            if self.ascadV2Type == "withPermIDs":
                labels = {}
                labels['alpha_output'] = to_categorical(Y_attack['alpha_mask'], num_classes=256)
                labels['beta_output'] = to_categorical(Y_attack['beta_mask'], num_classes=256)
                for i in range(16):
                    labels['sbox_' + str(i) + '_output'] = to_categorical(Y_attack['sbox_masked'][:, i], num_classes=256)
                for i in range(16):
                    labels['permind_' + str(i) + '_output'] = to_categorical(Y_attack['perm_index'][:, i], num_classes=16)
            elif self.ascadV2Type == "withoutPermIDs":
                labels = {}
                labels['alpha_output'] = to_categorical(Y_attack['alpha_mask'], num_classes=256)
                labels['beta_output'] = to_categorical(Y_attack['beta_mask'], num_classes=256)
                for i in range(16):
                    labels['sbox_' + str(i) + '_output'] = to_categorical(Y_attack['sbox_masked_with_perm'][:, i], num_classes=256)
            traces = X_attack[:self.testTracesNum, :]
            plaintext = Metadata_attack[:self.testTracesNum]
            key = Metadata_attack['key'][:self.testTracesNum]
            inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == 'h5' and self.datasetName == 'CHES2025':
            # TODO finish CHES_2025 challenge test dataset loading
            tmp = self.attackWindow.split('_')
            attackWindow = [int(tmp[0]), int(tmp[1])]
            with h5py.File(self.inputTracesFile) as in_file:
                traces = np.array(in_file['Attack_traces/traces'])[-self.testTracesNum:, attackWindow[0]:attackWindow[1]]
                traces = traces.reshape((traces.shape[0], traces.shape[1]))
                plaintext = np.array(in_file['Attack_traces/metadata'][:]['plaintext'][-self.testTracesNum:, self.targetByte])
                key = np.array(in_file['Attack_traces/metadata'][:]['key'][0])  # Get the real key here (note that attack key are fixed)
            labels = loadData.get_labels(plaintext, key, self.targetByte, self.leakageModel.value)
            labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
            inp_shape = (traces.shape[1], 1)

        return traces, labels, plaintext, key, inp_shape


    def fullTest(self):
        """
        Runs a full test of the model against the attack traces, calculating accuracy, predictions, and ranking curve
        :return: None
        """
        startTestTimer = time.time()  # Start full test timer

        # Create output directory for test results
        if self.outputDir is not None:
            ranking_root = os.path.join(self.outputDir, 'rank_dir')
            os.makedirs(ranking_root, exist_ok=True)
        else:
            ranking_root = os.path.join(self.inputModelDir, 'rank_dir')
            os.makedirs(ranking_root, exist_ok=True)

        # Load profiling and attack data and metadata
        X_attack, Y_attack, plaintext, key, inp_shape = self.load_data()

        # Workaround for small window resnets
        if self.padTo is not None:
            X_attack = np.pad(X_attack, [(0, 0), (0, self.padTo - X_attack.shape[1])], constant_values=0)
            inp_shape = (X_attack.shape[1], 1)

        if self.bulk:  # Bulk mode
            modelName = list()
            modelSize = list()
            modelMTD  = list()
            if self.datasetName != "ASCADv2":
                for modelPath in self.h5Files:
                    modelName.append(modelPath)
                    model = load_model(modelPath)
                    model.summary()
                    modelSize.append((count_params(model.trainable_weights)+count_params(model.non_trainable_weights)))
                    input_layer_shape = model.get_layer(index=0).input_shape
                    if isinstance(input_layer_shape, list):
                        input_layer_shape = input_layer_shape[0]
                    Reshaped_X_attack = loadData.sanity_check(input_layer_shape, X_attack)
                    preds = model.predict(Reshaped_X_attack)
                    score, acc = model.evaluate(Reshaped_X_attack, Y_attack, verbose=False)
                    max_trace_num = preds.shape[0]
                    mtd = key_rank.computeRankingCurve(preds, key, plaintext, self.targetByte, ranking_root,
                                                       self.leakageModel.value, max_trace_num, self.datasetName)
                    modelMTD.append(mtd)
                    print(modelSize[-1], modelMTD[-1])
                for modelPath2 in self.tfliteFiles:
                    # TODO bulk test of tflite models
                    exit(1)

            elif self.datasetName == "ASCADv2":
                # TODO bulk test of ASCADv2 models
                exit(1)

            # Export lists to csv format
            with open(os.path.join(ranking_root, 'bulk_test_results.csv'), 'w') as f:
                f.write('Model Name,Model Size,Mean Traces to Disclosure\n')
                for i in range(len(modelName)):
                    f.write(f"{modelName[i]},{modelSize[i]},{modelMTD[i]}\n")
        else:
            # Get input layer shape, perform sanity check, and reshape
            if self.modelType == ModelType.H5:
                input_layer_shape = self.model.get_layer(index=0).input_shape
                if isinstance(input_layer_shape, list):
                    input_layer_shape = input_layer_shape[0]
            elif self.modelType == ModelType.TFLITE:
                input_layer_shape = self.model.get_input_details()[0]['shape']
                input_layer_shape = (input_layer_shape[0], input_layer_shape[1], input_layer_shape[2])

            # Run tests --- For H5 (Keras), use built-in evaluate and predict functions to get predictions
            if self.modelType == ModelType.H5:
                self.model.summary()  # Print off summary of model
                if self.datasetName != "ASCADv2":
                    Reshaped_X_attack = loadData.sanity_check(input_layer_shape, X_attack)
                    if self.plaintextAttack:
                        preds = self.model.predict(Reshaped_X_attack)
                        # need to expand
                        plaintext = np.argmax(preds[1], axis=1)
                        preds = preds[0]
                        max_trace_num = preds.shape[0]
                    else:
                        preds = self.model.predict(Reshaped_X_attack)
                        score, acc = self.model.evaluate(Reshaped_X_attack, Y_attack, verbose=self.verbose)
                        print(f'[LOG] -- Test acc is: {acc:.4f}')
                        max_trace_num = preds.shape[0]
                    key_rank.computeRankingCurve(preds, key, plaintext, self.targetByte, ranking_root,
                                                 self.leakageModel.value, max_trace_num, self.datasetName,
                                                 self.plaintextAttack)

                elif self.datasetName == "ASCADv2":  # Set simulated key and test all 16 bytes
                    Reshaped_X_attack = loadData.sanity_check(input_layer_shape, X_attack)
                    preds = self.model.predict(Reshaped_X_attack)
                    testResults = self.model.evaluate(Reshaped_X_attack, Y_attack, verbose=self.verbose)

                    # Gather averaged key rank results
                    if self.ascadV2Type == "withPermIDs":
                        acc = testResults[-32:-16]
                        predictions_sbox = ascadTest.multilabel_predict(preds)
                    elif self.ascadV2Type == "withoutPermIDs":
                        acc = testResults[-16:]
                        predictions_sbox = ascadTest.multilabel_without_permind_predict(preds)
                    print('[LOG] -- Byte Accuracies:')
                    for i in range(16):
                        print(f"{i:2d}: {acc[i]:.4f} ", end="")
                        if (i + 1) % 4 == 0:
                            print("")
                    allKeyRankArrays = list()
                    for target_byte in range(16):  # See average code within full_ranks()
                        ranks_i = ascadTest.full_ranks(predictions_sbox[target_byte], X_attack, plaintext, 0,
                                                       self.testTracesNum, 1, target_byte, 1)
                        x_i = [ranks_i[i][0] for i in range(0, ranks_i.shape[0])]
                        y_i = [ranks_i[i][1] for i in range(0, ranks_i.shape[0])]
                        allKeyRankArrays.append((x_i, y_i))

                    # Create and export all plots, export additional ASCADv2-specific results:
                    key_rank.ascadV2RankingCurves(allKeyRankArrays, self.ascadV2Type,
                                                  os.path.join(self.outputDir, 'rank_dir'))
                    with open((os.path.join(self.outputDir, 'rank_dir', '16_key_rank_text.txt')), 'w') as f:
                        for i, byte in enumerate(allKeyRankArrays):
                            f.write(f"{i}: {byte[1]}\n")

            # For TFLite, need to set tensors and collect predictions the manual way
            elif self.modelType == ModelType.TFLITE:
                X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
                preds = [None] * self.testTracesNum  # Python list pre-allocation
                inputTensor = self.model.tensor(self.model.get_input_details()[0]['index'])
                acc = "TFLITE"
                # outputTensor = model.tensor(model.get_output_details()[0]['index'])  # no good way to use this
                for testTrace in range(X_attack.shape[0]):  # Equivalent to model.predict(X_attack)
                    inputTensor()[0, :, :] = X_attack[testTrace]
                    self.model.invoke()
                    preds.append(self.model.get_tensor(self.model.get_output_details()[0]['index'])[0])
                    print(f"\rProgress: {testTrace}/{X_attack.shape[0]} ({testTrace / X_attack.shape[0] * 100:.2f}%)",
                          end='')
                print(f"\rProgress: {X_attack.shape[0]}/{X_attack.shape[0]} (100.00%)")
                preds = np.array(preds)
                max_trace_num = min(self.testTracesNum, preds.shape[0])
                key_rank.computeRankingCurve(preds, key, plaintext, self.targetByte, ranking_root,
                                       self.leakageModel.value, max_trace_num, self.datasetName)

        endTestTimer = time.time()
        timeDelta = endTestTimer - startTestTimer
        print(f"Time to test {len(X_attack)} traces: {endTestTimer - startTestTimer:.2f} seconds")
        print(f"Time per trace: {(endTestTimer - startTestTimer) / len(X_attack):.6f} seconds")
        print('Exporting reproducibility statistics...')
        self.exportReproducibilityStats(timeDelta, acc)
        print('[LOG] ---- All done!')


    def setNewTestDir(self, newTestDir):
        """
        Set a new test directory for the SideChannelTester object.
        Recalculates 'inputModelFile', 'modelType' and 'model' then re-verifies parameters.
        Will exit if any of the required parameters are invalid.
        :param newTestDir: Top-level directory filepath to the new test directory (e.g.
        :return:
        """
        self.inputModelDir = newTestDir

        # Recalculate additional parameters
        # Store exact model file path and extension, then load model
        if os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.tflite')):
            self.inputModelFile = os.path.join(self.inputModelDir, 'model', 'best_model.tflite')
        elif os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.h5')):
            self.inputModelFile = os.path.join(self.inputModelDir, 'model', 'best_model.h5')
        if self.inputModelFile.split('.')[1].lower() == 'h5':
            self.modelType = ModelType.H5
            self.model = load_model(self.inputModelFile)
        elif self.inputModelFile.split('.')[1].lower() == 'tflite':
            self.modelType = ModelType.TFLITE
            self.model = tf.lite.Interpreter(model_path=self.inputModelFile)
            self.model.allocate_tensors()

        # Ensure that the required parameters are still valid
        self.verifyParameters()


    def exportReproducibilityStats(self, timeDelta, testAccuracy):
        """
        Export reproducibility statistics to a file in the output directory after tests are complete
        :param timeDelta: Floating point time delta representing the time it took to run the test
        :param testAccuracy: Test accuracy of the model against the traces
        :return: None
        """
        # Ensure output directory for test results was actually created
        ranking_root = os.path.join(self.outputDir, 'rank_dir')
        if not os.path.exists(ranking_root):
            print('[FATAL] -- Output directory for test results was not created after test.')
            exit(1)

        reprodOutputLogFile = os.path.join(ranking_root, f'test_{self.datasetName}_{date.today()}.log')
        reprodLog = list()
        reprodLog.append(f"{os.path.join(ranking_root, f'test_{self.datasetName}_{date.today()}.log')}")
        reprodLog.append(f"Test completed on {time.ctime(time.time())}")
        reprodLog.append(f"Test results saved to {os.path.join(self.outputDir, 'rank_dir')}")
        reprodLog.append(f"Testing script run with the following command:")
        reprodLog.append("python3 " + " ".join(sys.argv))
        reprodLog.append(f"Test took {timeDelta:.2f} seconds with {self.testTracesNum} traces")
        reprodLog.append(f"Test accuracy: {testAccuracy}")
        reprodLog.append(f" -------------- Current library versions: --------------")
        reprodLog.append(f"Python: {sys.version}")
        reprodLog.append(f"Tensorflow: {tf.__version__}")
        reprodLog.append(f"NVidia CUDA Runtime version: {build_info.build_info['cuda_version']}")
        reprodLog.append(f"NVidia CUDNN Runtime version: {build_info.build_info['cudnn_version']}")
        reprodLog.append(f" -------------- Current test object parameters: --------------")
        reprodLog.append(f"Model directory: {self.inputModelDir}")
        reprodLog.append(f"Model file: {self.inputModelFile}")
        reprodLog.append(f"Model type: {self.modelType.value}")
        reprodLog.append(f"Model hash (SHA256): {hashlib.sha256(open(self.inputModelFile, 'rb').read()).hexdigest()}")
        reprodLog.append(f"Dataset file: {self.inputTracesFile}")
        reprodLog.append(f"Dataset name: {self.datasetName}")
        reprodLog.append(f"Dataset file extension: {self.datasetFileExtension}")
        reprodLog.append(
            f"Dataset hash (SHA256): {hashlib.sha256(open(self.inputTracesFile, 'rb').read()).hexdigest()}")
        reprodLog.append(f"Test traces: {self.testTracesNum}")
        reprodLog.append(f"Attack window: {self.attackWindow}")
        reprodLog.append(f"Target byte: {self.targetByte}")
        reprodLog.append(f"Leakage model: {self.leakageModel.value}")
        reprodLog.append(f"Preprocessing: {self.preprocess}")
        reprodLog.append(f"Shifted: {self.shifted}")
        reprodLog.append(f"Verbose: {self.verbose}")
        reprodLog.append(f"")

        # Write each line to the log file
        with open(reprodOutputLogFile, 'w') as f:
            for line in reprodLog:
                f.write(f"{line}\n")


def parseArgs():
    """
    Parse command line arguments if run from command line
    :return: Namespace object containing parsed arguments (i.e. args = parser.parse_args(); args.input;)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_traces', help='Input traces to test the model')
    parser.add_argument('-m', '--input_model_dir', help='Input original model to test against')
    parser.add_argument('-o', '--output_dir', help='Output directory for the test results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Include for verbose output')
    parser.add_argument('-tb', '--target_byte', type=int, help='Target byte to attack (0-15)')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'HD', 'ID'}, help='Leakage model of the network')
    parser.add_argument('-aw', '--attack_window', help='Attack window (POI window) for the traces')
    parser.add_argument('-tn', '--test_traces', type=int, help='Number of traces to test against')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scaling'},
                        help='Preprocessing method (unused)')
    parser.add_argument('-sh', '--shifted', type=int, default=0,
                        help='Shift the attack window by this many traces (unused)')
    parser.add_argument('-chesAttack', '--chesAttack', choices={'same', 'cross'},
                        help="Attack setting for the CHES dataset")
    parser.add_argument('-ascadv2Type', '--ascadv2_type', choices={'withPermIDs', 'withoutPermIDs'},
                        help="For use with ASCADv2 only. Selects the type of MultiSCAResNet.")
    parser.add_argument('-plaintext', '--plaintext_attack', action='store_true', help='Plaintext attack mode')
    parser.add_argument('-bulk', '--bulk', action='store_true', help='Bulk model processing mode. '
                                                                     'Input Model Dir should contain many models at '
                                                                     'root of the directory (.h5)')
    parser.add_argument('-padTo', '--padTo', type=int, default=None,
                        help='Pad the attack window up to this width with zeroes')
    opts = parser.parse_args()
    return opts


# ------------------ Main ------------------
if __name__ == "__main__":
    cmdArgs = parseArgs()
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    sideChannelTester = SideChannelTester(cmdArgs.input_traces, cmdArgs.input_model_dir, cmdArgs.output_dir,
                                          cmdArgs.target_byte, cmdArgs.leakage_model, cmdArgs.attack_window,
                                          cmdArgs.test_traces, cmdArgs.preprocess, cmdArgs.shifted, cmdArgs.chesAttack,
                                          cmdArgs.ascadv2_type, cmdArgs.verbose, cmdArgs.plaintext_attack, cmdArgs.bulk,
                                          cmdArgs.padTo)
    sideChannelTester.fullTest()
