# openShapNPZ.py - Logan Reichling - Start 9/4/25 - UC DaSec
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
    shapValues = loadNpzFile(r"shap_restore_smaeshHWPostSyn_b2_2.npz")
    with open("shapValueList.txt", "w") as outFile:
        for i, value in enumerate(shapValues):
            print(f"{i}: {value:.2f}")
            outFile.write(f"{value:.2f}\n")

    # Create figures of data
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(0, 505), shapValues, label="Mean SHAP Value")
    # plt.tight_layout(rect=(-0.04,0.02,1,1))
    plt.tight_layout(pad=1.6)
    plt.margins(x=0)
    plt.xlabel("Timestamp")
    plt.ylabel("Mean |SHAP Value|")
    xTicks = [0, 125, 250, 375, 500]
    plt.xticks(xTicks, xTicks)
    yTicks = [0, 1000, 2000, 3000, 4000, 5000]
    plt.yticks(yTicks, ["0", "1000", "2000", "3000", "4000", "5000"])
    # plt.legend(loc='upper right')
    plt.savefig("./gen/smaesh_syn_hw_shap.pdf")
    plt.show()
    plt.clf()

