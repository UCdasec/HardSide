# HardSide

**The code is for research purpose only**

HardSide is a collection of tools and datasets used in the evaluation of pre-silicon simulated side-channel traces of cryptographic hardware designs with deep learning. In this repository, we provide the tools and scripts used to produce our HardSide simulated trace dataset and train deep learning SCA models over said traces. Scripts to perform testing and explainability analysis over the traces are also provided in this repository. 

![hardsideSystemModelFigure](https://github.com/user-attachments/assets/ad146129-4e53-42a4-8b72-068bd85ca46c)

Our simulation pipeline generally follows five main steps:
 * Select a hardware design to evaluate
 * Collect many Value Change Dump (VCD) files
 * Convert the VCDs into simulated traces
 * Train and test classifiers using partitions of the simulated traces
 * Evaluate side-channel leakage of said traces and perform root cause analysis

## Reference

When reporting results that use the code or datasets in this repository, please cite the paper below:

Logan Reichling, Phuc Mai, Muhib Khan, Kaiden Thomas, and Boyang Wang, "HardSide: Pre-Silicon Side-Channel Analysis on Hardware AES with Deep Learning," IEEE International Conference on Physical Assurance and Inspection of Electronics (**PAINE 2026**), Phoenix AZ, October 27-29, 2026 (_to be published_)

Our datasets and hardware AES source projects used in this study can be accessed through the link below:

https://mailuc-my.sharepoint.com/:f:/r/personal/wang2ba_ucmail_uc_edu/Documents/group/datasets_public/side_channel_pre_silicon_hardside_20251121?d=w427d77d4f9ac483fab5de72e33f53569&csf=1&web=1&e=eW2hIc

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (boyang.wang@uc.edu). We will be update the links as soon as we can. Thanks!

# Content

 * ```vivado_all_in_one_tofu_capture.py``` - Combines generation and processing steps of the simulated trace pipeline using Vivado and TOFU to create a final simulated trace dataset.
 * ```only_tofu_wrapper.py``` - Performs the simulated trace generation and final dataset creation with already pre-generated VCD files.
 * ```parse_VCD.py``` - Helper tool for VCD files to convert VCDs into a more readable format
 * ```/tools/``` - Contains accessories for generating figures.
 * ```/trainingAndShap/``` - Contains functionality to train, test, and perform explainability analysis.
 * ```/trainingAndShap/train.py``` - Trains a SCA neural network over traces
 * ```/trainingAndShap/test.py``` - Tests a SCA neural network over traces
 * ```/trainingAndShap/shapScores.py``` - Collects SHAP scores over a given SCA model
 * ```/trainingAndShap/tools/``` - Contains accessory files used by the training code.

# Requirements

Python 3.10 or greater is needed while Tensorflow 2.11.0 was used for all model training. CUDA 12.1 and CUDNN 8 are installed via Deb files from the NVidia Website. Instructions on Deb installation can be found [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). 

A Conda environment is used to collect the required packages. The ```environment.yml``` file included in the trainingAndShap subdirectory has the required packages listed and can be created new by conda via the following command ```conda env create --name hardside --file=environment.yml```. Minor modifications to the environment may be needed depending on the final environment. 

# Usage
The simulated trace pipeline requires Vivado 2023.2 (see [here](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html)) and TOFU (see [here](https://gitlab.lrz.de/tueisec/tofu/-/tree/master)) to generate synthetic traces. The Python script, vivado_all_in_one_tofu_capture.py, handles the interaction between the two automatically, but requires the design to be run first in simulation in order to generate the correct paths. The initial script parameters at the beginning of the file are straightforward to set for your own environment. These initial script parameters need to be changed before execution of the script. A special note about the batch size parameter, which may need to be tweaked depending on the hardware and design used, however in our experience for under 50k traces you can set the BATCH_SIZE to the TRACES_TO_COLLECT size (A memory leak exists in Vivado which will crash simulation if BATCH_SIZE is too low). The script should be compatible with both Windows and Linux based on the provided parameter, but peculiarities may exist in your setup that are not accounted for within the current version of the script. 

Model training and SHAP analysis code exists within the trainingAndShap folder. These scripts should be passed parameters through the command line. Please see example usage of these scripts below (where results will appear in the final model directory):
Training a ResNet-S model:
 - ```python3 train.py -i ./dataset/SMAesH_wo_endian_reverse_SyntheticHammingWeight_K1_50k.npz -o ./Result/SMAesH_wo_endian_reverse_HammingWeight/SMAesH_test_DLSCA_ResNetSingle_ID -tn 40000 -e 150 -tb 13 -lm ID -aw 0_406 -mt ResNetSingle -v```
Testing a model:
 - ```python3 test.py -i ./dataset/SMAesH_wo_endian_reverse_SyntheticHammingWeight_K1_50k.npz -m ./Result/SMAesH_wo_endian_reverse_HammingWeight/SMAesH_test_DLSCA_ResNetSingle_ID -tn 10000 -tb 13 -lm ID -aw 0_406 -v```
Calculating SHAP scores for that model:
 - ```python3 shapScores.py -i ./dataset/SMAesH_wo_endian_reverse_SyntheticHammingWeight_K1_50k.npz -m ./Result/SMAesH_wo_endian_reverse_HammingWeight/SMAesH_test_DLSCA_ResNetSingle_ID/model/best_model.h5 -tn 40000 -tt 500 -tb 13 -lm ID -aw 0_406 -mt ResNetSingle```

Additional tools and scripts used for plot generation are located within the base tools directory. 

# Contacts
Logan Reichling reichlln@mail.uc.edu

Phuc Mai maipd@mail.uc.edu

Boyang Wang boyang.wang@uc.edu
