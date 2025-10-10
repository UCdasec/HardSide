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
    shapValues = loadNpzFile(r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\shap_restore_smaeshHWPostSyn_b2_Cut5.npz")
    shapValues2 = loadNpzFile(r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\shap_restore_smaeshHWPostSyn_b2_Cut6.npz")
    # with open("shapValueList.txt", "w") as outFile:
    #     for i, value in enumerate(shapValues):
    #         print(f"{i}: {value:.2f}")
    #         outFile.write(f"{value:.2f}\n")

    # Create figures of data
    plt.rcParams.update({'font.size': 16})

    # plt.figure(figsize=(4, 3))
    start = 0
    end = 505
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.plot(np.arange(start, end), shapValues2[start:end], label="Cut 6")
    plt.plot(np.arange(start, end), shapValues[start:end], label="Cut 5", color='orange')
    plt.tight_layout(rect=(-0.05,0.02,1,1))
    #plt.tight_layout(pad=1.3)
    plt.margins(x=0.02)
    plt.xlabel("Timestamp")
    plt.ylabel("Mean |SHAP Value|")
    xTicks = [0, 125, 250, 375, 500]
    plt.xticks(xTicks, xTicks)
    yTicks = [0, 10000, 20000, 30000]
    plt.yticks(yTicks, ["0", "10k", "20k", "30k"])
    # plt.legend(loc='upper right', prop={'size': 11.5})

    # force order on labels
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels = ['Cut 5', 'Cut 6']
    by_label = dict(zip(labels, handles))
    ordered_handles = [by_label[label] for label in ordered_labels]
    ax.legend(ordered_handles, ordered_labels, loc='upper right', prop={'size': 11.5})

    plt.savefig("./gen/smaesh_syn_hw_shap_cut6.pdf")
    plt.show()
    plt.clf()

