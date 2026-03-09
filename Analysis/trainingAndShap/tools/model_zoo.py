# model_zoo.py - Logan Reichling - UC DaSec
# Note that some code is adapted from the ASCAD repository to train ASCAD v1/v2 models. See the original ASCAD repo
# at https://github.com/ANSSI-FR/ASCAD/ for more information and original source

import pdb
import tensorflow
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPool1D, Input, AveragePooling1D, Reshape, AveragePooling2D
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dropout
from tensorflow.keras.layers import Activation, GlobalAveragePooling1D, MaxPooling1D
from tensorflow.keras.layers import Add, add, multiply
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant

# CNN Best model (Original from ASCAD)
def cnn_best(input_shape, emb_size=256, classification=True, compile=True, multiGPU=False):
    """
    Traditional VGG-like CNN used for the majority of deep learning side-channel analysis
     * Effective on unmasked traces, shuffling, and first-order masked traces (depending on # of traces)
    :param input_shape: Input shape of the traces (e.g. (1000, 1) for 1000 samples with 1 channel)
    :param emb_size: Final embedding output size
    :param classification: True to add the final softmax classification layer, false to get direct Dense output
    :param compile: True to complete compile() step with chosen per-model optimizer, false to perform compile elsewhere
    :param multiGPU: True to detect and use multiple GPUs, scaling lr accordingly
    :return: Model (keras) object
    """
    if not multiGPU:
        inp = Input(shape=input_shape)
        # inp = Reshape((1000, 1))(inp)
        # Block 1
        x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(inp)
        x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
        # Block 2
        x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
        # Block 3
        x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
        # Block 4
        x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
        # Block 5
        x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
        x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
        # Classification block
        x = Flatten(name='block_flatten')(x)
        x = Dense(4096, activation='relu', name='block_fc1')(x)
        x = Dense(4096, activation='relu', name='block_fc2')(x)
        if classification:
            x = Dense(emb_size, activation='softmax', name='preds')(x)
            model = Model(inp, x, name='cnn_best')
            if compile:
                optimizer = RMSprop(lr=0.00001)
                model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return model
        else:
            return inp, x
    else:
        mirror = tensorflow.distribute.MirroredStrategy()
        print(f"Using {mirror.num_replicas_in_sync} GPUs")
        with mirror.scope():
            inp = Input(shape=input_shape)
            # Block 1
            x = Conv1D(64, 11, strides=2, activation='relu', padding='same', name='block1_conv1')(inp)
            x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
            # Block 2
            x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
            # Block 3
            x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
            # Block 4
            x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
            # Block 5
            x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
            x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
            # Classification block
            x = Flatten(name='block_flatten')(x)
            x = Dense(4096, activation='relu', name='block_fc1')(x)
            x = Dense(4096, activation='relu', name='block_fc2')(x)
            if classification:
                x = Dense(emb_size, activation='softmax', name='preds')(x)
                # Create model.
                model = Model(inp, x, name='cnn_best')
                if compile:
                    optimizer = RMSprop(lr=0.00001 + 0.00001 * (mirror.num_replicas_in_sync - 1))
                    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                return model
            else:
                return inp, x


# MLP Best model (Original from ASCAD)
def mlp_best(input_dim, emb_size=256, node=200, layer_nb=4):
    """
    Typical feed-forward MLP model used for simple deep learning side-channel analysis
     * Effective on unmasked traces and easy datasets. Fast to train
    :param input_dim: Input dimension of the traces (e.g. 1000 for 1000 samples)
    :param emb_size: Final embedding output size
    :param node: Number of nodes in each hidden layer
    :param layer_nb: Number of hidden layers
    :return: Model (keras) object
    """
    model = Sequential()
    model.add(Dense(node, input_dim=input_dim, activation='relu'))
    for i in range(layer_nb):
        model.add(Dense(node, activation='relu'))
    model.add(Dense(emb_size, activation='softmax'))
    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# CNN Best model (2D Layer version)
def cnn_best_fpga(input_shape, emb_size=256, classification=True):
    """
    Special CNN model for compatibility with 2D DNN accelerator. Same architecture as cnn_best but with Conv2D layers
    :param input_shape: Input shape of the traces (e.g. (1000, 1) for 1000 samples with 1 channel)
    :param emb_size: Final embedding output size
    :param classification: True to add the final softmax classification layer, false to get direct Dense output
    :return: Model (keras) object
    """
    inp = Input(shape=(*input_shape, 1))
    # Block 1
    x = Conv2D(64, (11, 1), strides=(2, 1), activation='relu', padding='same', name='block1_conv2')(inp)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block1_pool')(x)
    # x = BatchNormalization()(x)
    # Block 2
    x = Conv2D(128, (11, 1), activation='relu', padding='same', name='block2_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block2_pool')(x)
    # x = BatchNormalization()(x)
    # Block 3
    x = Conv2D(256, (11, 1), activation='relu', padding='same', name='block3_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block3_pool')(x)
    # x = BatchNormalization()(x)
    # Block 4
    x = Conv2D(512, (11, 1), activation='relu', padding='same', name='block4_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block4_pool')(x)
    # x = BatchNormalization()(x)
    # Block 5
    x = Conv2D(512, (11, 1), activation='relu', padding='same', name='block5_conv2')(x)
    x = AveragePooling2D((2, 1), strides=(2, 1), name='block5_pool')(x)
    # x = BatchNormalization()(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        optimizer = RMSprop(lr=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    else:
        # embeddings = x
        x = Dense(emb_size, kernel_initializer="he_normal")(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        return model


# --------------------------- ResNet components start ---------------------------
# Resnet layer sub-function of ResNetSCA
def resnet_layer(inputs, num_filters=16, kernel_size=11, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True):
    conv = Conv1D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


# 2D ResNet Layer
def resnet_layer_2D(inputs, num_filters=16, kernel_size=11, strides=1, activation='relu', batch_normalization=True,
                 conv_first=True):
    conv = Conv2D(num_filters, kernel_size=(kernel_size,1), strides=(strides), padding='same',
                  kernel_initializer='he_normal')
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


# Branch of ResNetSCA that predict the multiplicative mask alpha
def alpha_branch(x):
    x = Dense(1024, activation='relu', name='fc1_alpha')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='alpha_output')(x)
    return x


# Branch of ResNetSCA that predict the additive mask beta
def beta_branch(x):
    x = Dense(1024, activation='relu', name='fc1_beta')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="softmax", name='beta_output')(x)
    return x


# Branch of ResNetSCA that predict the masked sbox output
def sbox_branch(x, i, emb_size=256):
    x = Dense(1024, activation='relu', name='fc1_sbox_' + str(i))(x)
    x = BatchNormalization()(x)
    x = Dense(emb_size, activation="softmax", name='sbox_' + str(i) + '_output')(x)
    return x


# Branch of ResNetSCA that predict the permutation indices
def permind_branch(x, i):
    x = Dense(1024, activation='relu', name='fc1_pemind_' + str(i))(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation="softmax", name='permind_' + str(i) + '_output')(x)
    return x


# 2D ResNet version
def resnet_v1_2d(input_shape, depth, num_classes=256, without_permind=0):
    if (depth - 1) % 18 != 0:
        raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = Input(shape=input_shape)
    x = resnet_layer_2D(inputs=inputs)
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer_2D(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer_2D(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer_2D(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        if num_filters < 256:
            num_filters *= 2
    x = AveragePooling2D(pool_size=(4, 1))(x)
    x = Flatten()(x)
    x_alpha = alpha_branch(x)
    x_beta = beta_branch(x)
    x_sbox_l = []
    x_permind_l = []
    for i in range(16):
        x_sbox_l.append(sbox_branch(x, i))
        x_permind_l.append(permind_branch(x, i))
    if without_permind != 1:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l + x_permind_l, name='extract_resnet')
    else:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l, name='extract_resnet_without_permind')
    optimizer = Adam()  # Default LR is 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Produces the ResNetSCA architecture from the ASCADv2 paper.
# If without_permind option is set to 1, the ResNetSCA model is built without permindices branch
def resnet_v1(input_shape, depth, num_classes=256, without_permind=0):
    if (depth - 1) % 18 != 0:
        raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        if num_filters < 256:
            num_filters *= 2
    x = AveragePooling1D(pool_size=4)(x)
    x = Flatten()(x)
    x_alpha = alpha_branch(x)
    x_beta = beta_branch(x)
    x_sbox_l = []
    x_permind_l = []
    for i in range(16):
        x_sbox_l.append(sbox_branch(x, i))
        x_permind_l.append(permind_branch(x, i))
    if without_permind != 1:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l + x_permind_l, name='extract_resnet')
    else:
        model = Model(inputs, [x_alpha, x_beta] + x_sbox_l, name='extract_resnet_without_permind')
    optimizer = Adam()  # Default LR is 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# Single branch ResNetSCA model modification
def resnetSingle(input_shape, depth, emb_size=256):
    """
    Our custom ResNetSCA model with a single output for the SBox prediction and modified AveragePooling1D layer.
    :param input_shape: Input shape of the traces (e.g. (1000, 1) for 1000 samples with 1 channel)
    :param depth: Number of layers in the ResNet (should be 18n+1, e.g. 19, 37, 55 ...)
    :param emb_size: Final embedding output size
    :return: Model (keras) object
    """
    if (depth - 1) % 18 != 0:
        raise ValueError('depth should be 18n+1 (eg 19, 37, 55 ...)')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        if num_filters < 256:
            num_filters *= 2
    x = AveragePooling1D(pool_size=2)(x)  # Slight modification to average pooling
    x = Flatten()(x)
    x_sbox = sbox_branch(x, 0, emb_size=emb_size)
    model = Model(inputs, x_sbox, name='resnet_single')
    optimizer = Adam()  # Default LR is 0.001
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model



