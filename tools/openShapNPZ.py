# openShapNPZ.py - Logan Reichling - Start 9/4/25 - UC DaSec
# Opens a NPZ relating to SHAP scores, exports to text, and regenerates plots

import numpy as np
import math
import matplotlib.pyplot as plt


def loadNpzFile(file_name):
    """
    loadNpzFile takes a file name and loads the numpy array from the file.
    :param section: Section of npz file to load
    :param file_name: The name of the file to load the numpy array from.
    :return: The numpy array loaded from the file.
    """
    try:
        with np.load(file_name) as dataArray:
            print(f"Available Sections {dataArray.files}")
            return dataArray['y']
    except (TypeError, ValueError) as e:
        print(f"Non-NPZ file format or unrecognized NPZ dataset")
        return None


if __name__ == "__main__":
    shapValues = loadNpzFile(r"C:\Users\Logan Reichling\Desktop\SHAP scores for Figure Generation\shap_restore_smaeshHW_b2.npz")
    # shapValues2 = loadNpzFile(r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\shap_restore_smaeshHWPostSyn_b2_Cut6_new.npz")
    # with open("shapValueList.txt", "w") as outFile:
    #     for i, value in enumerate(shapValues):
    #         print(f"{i}: {value:.2f}")
    #         outFile.write(f"{value:.2f}\n")

    # Create figures of data
    plt.rcParams.update({'font.size': 16})

    # plt.figure(figsize=(4, 3))
    start = 0
    end = 406
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.plot(np.arange(start,end), shapValues[start:end], )
    #plt.plot(np.arange(start, end), shapValues2[start:end], label="Iter. 6", )
    #plt.plot(np.arange(start, end), shapValues[start:end], label="Iter. 5", color='orange', alpha=0.8)
    plt.tight_layout(rect=(-0.06,0.02,1,1))
    #plt.tight_layout(pad=1.3)
    plt.margins(x=0.02)
    plt.xlabel("Timestamp")
    plt.ylabel("Mean |SHAP Value|")
    xTicks = [0, 100, 200, 300, 400]
    plt.xticks(xTicks, xTicks)
    yTicks = [0, 1000, 2000, 3000, 4000, 5000]
    plt.yticks(yTicks, ["0", "1k", "2k", "3k", "4k", "5k"])
    # plt.legend(loc='upper right', prop={'size': 11.5})

    # force order on labels
    # handles, labels = ax.get_legend_handles_labels()
    # ordered_labels = ['Iter. 5', 'Iter. 6']
    # by_label = dict(zip(labels, handles))
    # ordered_handles = [by_label[label] for label in ordered_labels]
    # ax.legend(ordered_handles, ordered_labels, loc='upper right', prop={'size': 11.5})

    plt.savefig("./gen/smaesh_rtl_hw_shap.pdf")
    plt.show()
    plt.clf()

