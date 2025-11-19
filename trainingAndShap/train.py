# train.py - Start 1/16/2024 - UC DASEC - Logan Reichling

import argparse
import hashlib
import os
import re
import sys
import time
from datetime import date
import h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.python.platform import build_info
tf.get_logger().setLevel('ERROR')
import tools.loadData as loadData
import tools.model_zoo as model_zoo
from tools.SideChannelConstants import SideChannelConstants, LeakageModel


class SideChannelTrainer:
    """
    Class to train a NN side channel model with a given set of parameters
    """
    def __init__(self, inputTracesFile: str, inputModelDir: str, outputDir: str, trainTraceNum: int, epochs: int,
                 targetByte: int, leakageModel: str, attackWindow: str, multiGPU: bool, modelType: str,
                 ascadV2Type: str, twoDMode: bool, verbose: bool = False, preprocess: str = "", padTo: int = None):
        """
        Initialize a SideChannelTrainer object by which to train new SCA models with traces
        :param inputTracesFile: Direct filepath to the .npz file containing the training traces
        :param inputModelDir:   Directory filepath to a pre-trained model to train further
        :param outputDir:       Directory filepath to the top level output directory for the trained model
        :param targetByte:      Byte number of the key to be tested (0-15)
        :param epochs:          Number of epochs to train the model
        :param leakageModel:    String representing the leakage model to be used ('HW' or 'ID')
        :param attackWindow:    String representing the attack window to be used (e.g. '1200_2200')
        :param trainTraceNum:   Number of traces to train with (e.g. 40000)
        :param modelType:
        :param multiGPU:        Boolean to turn on multi GPU training (experimental)
        :param ascadV2Type      For use with ASCADv2 dataset only. Selects type of MultiSCAResNet model to utilize.
        :param verbose:         Boolean representing whether to print optional, verbose output
        :param twoDMode
        :param preprocess:      String representing the preprocessing to be used ('', 'norm', or 'scaling')
        """
        self.inputTracesFile = inputTracesFile
        self.inputModelDir = inputModelDir
        self.model = None
        self.outputDir = outputDir
        self.trainTracesNum = trainTraceNum
        self.epochs = epochs
        self.targetByte = targetByte
        self.attackWindow = attackWindow
        self.multiGPU = multiGPU
        self.modelType = modelType
        self.ascadV2Type = ascadV2Type
        self.preprocess = preprocess
        self.verbose = verbose
        self.twoDMode = twoDMode
        self.padTo = padTo

        # Calculate additional parameters
        if leakageModel == 'HW':
            self.leakageModel = LeakageModel.HW
        elif leakageModel == 'ID':
            self.leakageModel = LeakageModel.ID
        elif leakageModel == 'HD':
            self.leakageModel = LeakageModel.HD
        else:  # leakageModel == 'NONE':
            # self.leakageModel = LeakageModel.NONE
            print("[FATAL] -- Leakage model not supported currently, exiting...")
            exit(1)

        # Extract testing dataset name for later metadata use (e.g. X1_K1_200k)
        _, fileName = os.path.split(self.inputTracesFile)
        self.datasetName = fileName.split('.')[0]
        self.datasetFileExtension = fileName.split('.')[1]

        # For 'additional training mode', store exact model file path and extension, then load model
        if self.inputModelDir is not None:
            if os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.tflite')):
                print('[FATAL] -- TFLite models are not supported for additional training (at this point).')
                exit(1)
            elif os.path.exists(os.path.join(self.inputModelDir, 'model', 'best_model.h5')):
                self.inputModelFile = os.path.join(self.inputModelDir, 'model', 'best_model.h5')
                self.model = load_model(self.inputModelFile)

        # Ensure that the required parameters are valid
        self.verifyParameters()


    def verifyParameters(self):
        """
        Verify that the training parameters are valid. EXIT if invalid.
        :return: None
        """
        exitFlag = False
        if not os.path.exists(self.inputTracesFile):
            print('[FATAL] -- Input traces file does not exist.')
            exitFlag = True
        if self.datasetFileExtension != 'npz' and self.datasetFileExtension != 'h5':
            print(f'[FATAL] -- Input traces file {self.datasetName} must be a .npz or .h5 file.')
            exitFlag = True
        datasetMatch = re.compile(SideChannelConstants.getDatasetNamingConvention())
        if (datasetMatch.match(self.datasetName) is None and
                self.datasetName != 'ASCAD' and self.datasetName != 'ASCADv2' and self.datasetName != 'CHES' and
                self.datasetName[:9] != 'Synthetic' and self.datasetName[:6] != 'SMAesH'):
            print(f'[WARN] -- Input traces "{self.datasetName}" file unrecognized. Modify python file if needed.')
            # exitFlag = True
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
        if self.modelType not in ["CNN", "MLP", "ResNetSingle", "ASCADv2"]:
            print('[FATAL] -- Model type must be either "CNN", "MLP", "ResNetSingle", or "ASCADv2".')
            exitFlag = True
        if self.leakageModel is None or self.leakageModel not in LeakageModel:
            print('[FATAL] -- Leakage model must be either "HW", "HD", or "ID".')
            exitFlag = True
        if self.trainTracesNum < 0:
            print('[FATAL] -- Test trace number must be greater than 0.')
            exitFlag = True
        if self.preprocess not in ['', 'norm', 'scaling']:
            print('[FATAL] -- Preprocess must be either norm or scaling (if provided).')
            exitFlag = True
        if exitFlag:
            exit(1)  # Exit if any of the above parameters are invalid


    def loadData(self):
        """
        Class function designed for use with an initialized SideChannelTrainer object.
        Load profiling and training data from the given npz traces file.
        :return: Traces (x values for model, power value sampled),
            labels (y values for model, intermediate AES representation),
            plaintext: original plaintext string,
            key: 128-bit key in hex string,
            inp_shape: Shape of x values traces to compare against model
        """
        if self.datasetFileExtension == 'npz':  # Our format
            whole_pack = np.load(self.inputTracesFile)
            traces, plaintext, key = loadData.loadDataTraining(whole_pack, self.attackWindow, self.preprocess,
                                                               self.trainTracesNum, shifted=0)
            labels = loadData.get_labels(plaintext, key, self.targetByte, self.leakageModel.value)
            if self.leakageModel != LeakageModel.NONE:
                labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
            elif self.modelType == "ResNetP":
                labels['sbox_0_output'] = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
                labels['plaintext_output'] = to_categorical(plaintext, self.leakageModel.getEmbeddingSize(self.leakageModel))
            inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == 'h5' and self.datasetName == 'ASCAD':  # ASCAD format:
            print("[NOTICE] -- Loaded ASCADv1 database ignores user-set window, target byte, and preprocess")
            self.attackWindow = "0_700"
            self.targetByte = 2
            self.preprocess = ""
            with h5py.File(self.inputTracesFile) as in_file:
                traces = np.array(in_file['Profiling_traces/traces'])[:self.trainTracesNum, :]
                labels = np.array(in_file['Profiling_traces/labels'], dtype='uint8')[:self.trainTracesNum]
                plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:self.trainTracesNum]
                key = np.array(in_file['Profiling_traces/metadata'])['key'][0]
                labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
                inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == "h5" and self.datasetName == 'ASCADv2':
            print("[NOTICE] -- Loaded ASCADv2 database and assuming MultiSCAResNet model...")
            print("[WARNING] -- Ignoring user-set window, target byte, preprocess, trace_num, epochs, and batch size")
            self.attackWindow = "0_15000"
            self.targetByte = "1-16"  # Now used for reporting purposes only
            self.preprocess = ""
            self.trainTracesNum = 420000
            self.epochs = 60
            with h5py.File(self.inputTracesFile) as in_file:
                traces = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)[:self.trainTracesNum, :]
                tempLabels = np.array(in_file['Profiling_traces/labels'])[:self.trainTracesNum]
                plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:self.trainTracesNum]
                key = np.array(in_file['Profiling_traces/metadata'])['key'][0]
                if self.ascadV2Type == "withPermIDs":
                    labels = {'alpha_output': to_categorical(tempLabels['alpha_mask'], num_classes=256),
                              'beta_output': to_categorical(tempLabels['beta_mask'], num_classes=256)}
                    for i in range(16):
                        labels['sbox_' + str(i) + '_output'] = to_categorical(tempLabels['sbox_masked'][:, i],
                                                                              num_classes=256)
                    for i in range(16):
                        labels['permind_' + str(i) + '_output'] = to_categorical(tempLabels['perm_index'][:, i],
                                                                                 num_classes=16)
                elif self.ascadV2Type == "withoutPermIDs":
                    labels = {'alpha_output': to_categorical(tempLabels['alpha_mask'], num_classes=256),
                              'beta_output': to_categorical(tempLabels['beta_mask'], num_classes=256)}
                    for i in range(16):
                        labels['sbox_' + str(i) + '_output'] = to_categorical(tempLabels['sbox_masked_with_perm'][:, i],
                                                                              num_classes=256)
                inp_shape = (traces.shape[1], 1)

        elif self.datasetFileExtension == 'h5' and self.datasetName == 'CHES':  # CHES format:
            print("[Notice] -- Loaded CHES database for training")
            with h5py.File(self.inputTracesFile) as in_file:
                # Set attack window:
                tmp = self.attackWindow.split('_')
                attackWindow = [int(tmp[0]), int(tmp[1])]
                traces = np.array(in_file['Profiling_traces/traces'])[:self.trainTracesNum,
                         attackWindow[0]:attackWindow[1]]
                plaintext = np.array(in_file['Profiling_traces/metadata'])['plaintext'][:self.trainTracesNum]
                plaintext = plaintext.astype(np.uint8)
                key = np.array(in_file['Profiling_traces/metadata'])['key'][0]  # Just get first key since its fixed
                key = key.astype(np.uint8)
                labels = loadData.get_labels(plaintext, key, self.targetByte, self.leakageModel.value)
                labels = to_categorical(labels, self.leakageModel.getEmbeddingSize(self.leakageModel))
                inp_shape = (traces.shape[1], 1)
        return traces, labels, plaintext, key, inp_shape


    def createLossAndAccuracyGraphs(self, exportDir, history):
        """
        Creates a graph of the loss function over time and saves it to the given export directory
        :param history: History object from model.fit()
        :param exportDir: Directory in which to export the figures
        :return: None
        """
        plt.rcParams.update({'font.size': 22})
        plt.rc('legend', fontsize=20)
        # plt.title(f'Model Training Loss w/ {self.datasetName} Dataset')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        x_ticks = np.arange(0, len(history.history['loss']), 6)  # [0, 30, 60, 90, 120, 150]
        plt.xticks(x_ticks, np.round(x_ticks, 0).astype(int))
        ax = plt.gca()
        ax.set_xlim(0 - 1, len(history.history['loss']) + 1)
        plt.margins(x=0)  # Tight plot bounds (no starting and ending whitespace)
        figure = plt.gcf()
        figure.set_size_inches(11, 8)
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(exportDir, 'loss.png'))
        plt.savefig(os.path.join(exportDir, 'loss.pdf'))
        plt.clf()

        plt.rcParams.update({'font.size': 22})
        plt.rc('legend', fontsize=20)
        # plt.title(f'Model Accuracy w/ {self.datasetName} Dataset')
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        x_ticks = np.arange(0, len(history.history['loss']), 6)  # [0, 30, 60, 90, 120, 150]
        plt.xticks(x_ticks, np.round(x_ticks, 0).astype(int))
        ax = plt.gca()
        ax.set_xlim(0 - 1, len(history.history['accuracy']) + 1)
        plt.margins(x=0)  # Tight plot bounds (no starting and ending whitespace)
        figure = plt.gcf()
        figure.set_size_inches(11, 8)
        plt.legend(['Accuracy', 'Val. Accuracy'], loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(exportDir, 'accuracy.png'))
        plt.savefig(os.path.join(exportDir, 'accuracy.pdf'))
        plt.clf()


    def exportReproducibilityStats(self, timeDelta, maxTrainLoss, modelFilePath):
        """
        Export reproducibility statistics to a file in the output directory after training is complete
        :param timeDelta: Floating point time delta representing the time it took to complete training
        :param maxTrainLoss: Final loss value from the training
        :param modelFilePath: File path to the trained model
        :return: None
        """
        # Ensure output directory for trained model was actually created
        modelOutputDir = os.path.join(self.outputDir, 'model')
        if not os.path.exists(modelOutputDir):
            print('[FATAL] -- Output directory for trained model was not created for training.')
            exit(1)
        if not os.path.exists(modelFilePath):
            print('[FATAL] -- Output trained model file was not created.')
            exit(1)

        reprodOutputLogFile = os.path.join(modelOutputDir, f'train_{self.datasetName}_{date.today()}.log')
        reprodLog = list()
        reprodLog.append(f"{os.path.join(modelOutputDir, f'train_{self.datasetName}_{date.today()}.log')}")
        if self.model is not None:
            reprodLog.append(f"Additional training completed on {time.ctime(time.time())}")
            reprodLog.append(f"Trained model loaded from {self.inputModelFile}")
        else:
            reprodLog.append(f"Training completed on {time.ctime(time.time())}")
        reprodLog.append(f"Trained model saved to {modelOutputDir}")
        reprodLog.append(f"Training script run with the following command:")
        reprodLog.append("python3 " + " ".join(sys.argv))
        reprodLog.append(f"Training took {timeDelta:.2f} seconds with {self.epochs} epochs")
        reprodLog.append(f"Final loss: {maxTrainLoss}")
        reprodLog.append(f" -------------- Current library versions: --------------")
        reprodLog.append(f"Python: {sys.version}")
        reprodLog.append(f"Tensorflow: {tf.__version__}")
        reprodLog.append(f"NVidia CUDA Runtime version: {build_info.build_info['cuda_version']}")
        reprodLog.append(f"NVidia CUDNN Runtime version: {build_info.build_info['cudnn_version']}")
        reprodLog.append(f" -------------- Current train object parameters: --------------")
        reprodLog.append(f"Model file export path: {modelFilePath}")
        reprodLog.append(f"Model hash (SHA256): {hashlib.sha256(open(modelFilePath, 'rb').read()).hexdigest()}")
        reprodLog.append(f"Dataset file: {self.inputTracesFile}")
        reprodLog.append(f"Dataset name: {self.datasetName}")
        reprodLog.append(f"Dataset file extension: {self.datasetFileExtension}")
        reprodLog.append(
            f"Dataset hash (SHA256): {hashlib.sha256(open(self.inputTracesFile, 'rb').read()).hexdigest()}")
        reprodLog.append(f"Train traces: {self.trainTracesNum}")
        reprodLog.append(f"Attack window: {self.attackWindow}")
        reprodLog.append(f"Target byte: {self.targetByte}")
        reprodLog.append(f"Leakage model: {self.leakageModel.value}")
        reprodLog.append(f"Preprocessing: {self.preprocess}")
        reprodLog.append(f"Verbose: {self.verbose}")
        reprodLog.append(f"")

        # Write each line to the log file
        with open(reprodOutputLogFile, 'w') as f:
            for line in reprodLog:
                f.write(f"{line}\n")


    def getModel(self, inputShape, embeddingSize, classification, multiGPU):
        """
        Returns the model object given the input shape, embedding size, and classification type
        :param inputShape: Input shape of the model (e.g. (700, 1) for 700 samples, 1 channel)
        :param embeddingSize: Output embedding size for the model (e.g. 9 for HW, 256 for ID)
        :param classification: True if the model is for classification, False if not
        :param multiGPU: Optional boolean to turn on multi GPU training (experimental)
        :return: Keras model object created from the model zoo
        """
        createdModel = None
        if self.modelType == "CNN":
            if self.twoDMode:
                print("[LOG] -- Using special 2D layer CNN model for DNN accelerators...")
                createdModel = model_zoo.cnn_best_fpga(inputShape, emb_size=embeddingSize,
                                                       classification=classification)
            else:
                print("[LOG] -- Using normal CNN model...")
                createdModel = model_zoo.cnn_best(inputShape, emb_size=embeddingSize, classification=classification,
                                                  multiGPU=multiGPU)
            return createdModel
        elif self.modelType == "MLP":
            print("[LOG] -- Using MLP model...")
            createdModel = model_zoo.mlp_best(inputShape[0], emb_size=embeddingSize)
            return createdModel
        elif self.modelType == "ResNetSingle":
            print("[LOG] -- Using ResNetSingle model...")
            createdModel = model_zoo.resnetSingle(inputShape, 19, emb_size=embeddingSize)
            return createdModel
        elif self.modelType == "ASCADv2":
            if self.twoDMode:
                print("[LOG] -- Using special 2D layer model for DNN accelerators...")
                if self.ascadV2Type == "withPermIDs":
                    createdModel = model_zoo.resnet_v1_2d((15000, 1, 1), 19)
                elif self.ascadV2Type == "withoutPermIDs":
                    createdModel = model_zoo.resnet_v1_2d((15000, 1, 1), 19, without_permind=1)
            else:
                if self.ascadV2Type == "withPermIDs":
                    createdModel = model_zoo.resnet_v1((15000, 1), 19)
                elif self.ascadV2Type == "withoutPermIDs":
                    createdModel = model_zoo.resnet_v1((15000, 1), 19, without_permind=1)
            return createdModel
        else:
            print("[FATAL] -- Invalid model type specified!")
            exit(1)


    def getHyperparameters(self, modelOutputFilePath):
        """
        Returns the proper hyperparameters {'callbacks', 'batchSize', 'valSplit'} given the model type
        :param modelOutputFilePath: File path to save the resultant h5 model
        :return: Dict object containing relevant hyperparameters per model type {'callbacks', 'batchSize', 'valSplit'}
        """
        if self.modelType in ['CNN', 'MLP']:
            checkpointer = ModelCheckpoint(modelOutputFilePath, monitor='val_accuracy', verbose=self.verbose,
                                           mode='max', save_best_only=True)
            return {"callbacks": [checkpointer], "batchSize": 100, "valSplit": 0.1}
        elif self.modelType in ['ResNetSingle']:
            checkpointer = ModelCheckpoint(modelOutputFilePath, monitor='val_accuracy',
                                           verbose=self.verbose, mode='max', save_best_only=True)
            return {"callbacks": [checkpointer], "batchSize": 64, "valSplit": 0.1}
        elif self.modelType == "ASCADv2":
            # Replicate ASCADv2 model training
            checkpointer = ModelCheckpoint(modelOutputFilePath)
            earlyStopper = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            return {"callbacks": [checkpointer, earlyStopper], "batchSize": 64, "valSplit": 0.05}
        else:
            print("[FATAL] -- Invalid model type specified!")
            exit(1)


    def mainTrain(self):
        """
        Trains new model against the training traces with the given parameters
        :return: None
        """
        startTrainTime = time.time()

        # Create output directory and output file path
        modelDir = os.path.join(self.outputDir, 'model')
        os.makedirs(modelDir, exist_ok=True)
        modelSaveFile = os.path.join(modelDir, 'best_model.h5')

        # Load data and reshape data
        X_profiling, Y_profiling, plaintext, key, input_shape = self.loadData()

        # Workaround for small window resnets
        if self.padTo is not None:
            X_attack = np.pad(X_profiling, [(0, 0), (0, self.padTo - X_profiling.shape[1])], constant_values=0)
            inp_shape = (X_attack.shape[1], 1)

        # Get model architecture
        if self.model is None:
            model = self.getModel(input_shape, self.leakageModel.getEmbeddingSize(self.leakageModel),
                                  classification=True, multiGPU=self.multiGPU)
        else:
            model = self.model
        hyperparameters = self.getHyperparameters(modelSaveFile)
        model.summary()

        # Get the input layer shape and reshape training data to match model type
        input_layer_shape = model.get_layer(index=0).input_shape
        if isinstance(input_layer_shape, list):
            input_layer_shape = input_layer_shape[0]
        print('Input shape', input_layer_shape)
        Reshaped_X_profiling = loadData.sanity_check(input_layer_shape, X_profiling)

        # Train model
        history = model.fit(x=Reshaped_X_profiling, y=Y_profiling, validation_split=hyperparameters['valSplit'],
                            batch_size=hyperparameters['batchSize'], verbose=self.verbose, epochs=self.epochs,
                            shuffle=True, callbacks=hyperparameters['callbacks'])

        endTrainTime = time.time()
        timeDelta = endTrainTime - startTrainTime
        print(f'Time to train {self.epochs} epochs: {timeDelta:.4f} seconds')
        print(f"Time per epoch: {timeDelta / self.epochs:.4f} seconds")

        # Export additional statistics and reproducibility information
        self.exportReproducibilityStats(timeDelta, history.history['loss'][-1], modelSaveFile)
        self.createLossAndAccuracyGraphs(modelDir, history)
        print('[LOG] -- All done!')


def parseArgs():
    """
    Parse command line arguments if run from command line
    :return: Namespace object containing parsed arguments (i.e. args = parser.parse_args(); args.input;)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_traces', help='Input traces to train the model')
    parser.add_argument('-m', '--inputModelDir', help='Input model directory to train further')
    parser.add_argument('-o', '--output_dir', help='Output directory for the trained model')
    parser.add_argument('-tn', '--train_traces', type=int, help='Number of traces to train the model')
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train the model')
    parser.add_argument('-tb', '--target_byte', type=int, help='Target byte to attack (0-15)')
    parser.add_argument('-lm', '--leakage_model', choices={'HW', 'HD', 'ID'}, help='Leakage model of the network')
    parser.add_argument('-aw', '--attack_window', help='Attack window (POI window) for the traces')
    parser.add_argument('-multiGPU', '--multi_gpu', action='store_true', default=False,
                        help='Include for distributed training')
    parser.add_argument("-mt", "--model_type", choices={"CNN", "MLP", "ResNetSingle", "ASCADv2"},
                        help="Model type for training.")
    parser.add_argument('-ascadv2Type', '--ascadv2_type', choices={'withPermIDs', 'withoutPermIDs'},
                        help="For use with ASCADv2 only. Selects the type of MultiSCAResNet.")
    parser.add_argument('-fpga', '--fpga', action='store_true', help='Include for 2D model layers')
    parser.add_argument('-v', '--verbose', action='store_true', help='Include for verbose output')
    parser.add_argument('-pp', '--preprocess', default='', choices={'', 'norm', 'scaling'},
                        help='Preprocessing method (unused)')
    opts = parser.parse_args()
    return opts


# Main function to run the training script
if __name__ == "__main__":
    cmdArgs = parseArgs()
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    sideChannelTrainer = SideChannelTrainer(cmdArgs.input_traces, cmdArgs.inputModelDir, cmdArgs.output_dir,
                                            cmdArgs.train_traces, cmdArgs.epochs, cmdArgs.target_byte,
                                            cmdArgs.leakage_model, cmdArgs.attack_window, cmdArgs.multi_gpu,
                                            cmdArgs.model_type, cmdArgs.ascadv2_type, cmdArgs.fpga, cmdArgs.verbose,
                                            cmdArgs.preprocess)
    sideChannelTrainer.mainTrain()
