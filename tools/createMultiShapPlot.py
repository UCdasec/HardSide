# createMultiShapPlot.py
import numpy as np
import matplotlib.pyplot as plt


def loadShapNpzFile(file_name):
    """
    loadNpzFile takes a file name and loads the numpy array from the file.
    :param section: Section of npz file to load
    :param file_name: The name of the file to load the numpy array from.
    :return: The numpy array loaded from the file.
    """
    try:
        with np.load(file_name) as dataArray:
            return dataArray['y']
    except (TypeError, ValueError) as e:
        print(f"Non-NPZ file format or unrecognized NPZ dataset")
        return None


# Main
if __name__ == "__main__":
    # Parameters
    selectedTrace = 0
    poiZoom = (0, 505)
    xStepSize = 105
    yStepSize = 500
    datasets = [
        # r"C:\Users\Logan Reichling\Desktop\shap_restore_smaeshHWPostSyn_b2_orig.npz",
        r"C:\Users\Logan Reichling\Desktop\shap_restore_smaeshHWPostSyn_b2_Cut6.npz"
    ]


    selectedShapPlotValues = list()
    for i, datasetPath in enumerate(datasets):
        shapValues = loadShapNpzFile(datasetPath)
        selectedShapPlotValues.append(shapValues)

    # Start plot...
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(4, 3))

    # plt.axhline(y=0, color='k', linestyle='dashed', linewidth=0.5)  # Line at 0

    colorList = ['b', 'g', 'r', 'orange', 'purple', 'teal', 'olivedrab']
    for i, trace in enumerate(selectedShapPlotValues):
        zoomed_data = trace[poiZoom[0]:poiZoom[1]]
        if i == 0:
            label = "Orig."
        else:
            label = f"Cut 6"
        plt.plot(np.arange(poiZoom[0], poiZoom[1]), zoomed_data, label=label, color=colorList[i])

    plt.margins(x=0, y=0.03)  # Tight plot bounds (no starting and ending whitespace)
    plt.legend(loc="upper left", prop={'size': 11.5})

    plt.tight_layout(rect=[-0.12, 0.02, 1, 1])  # (left, bottom, right, top)

    # plt.title(f"{fileName} Trace POI Window index from {poiZoom[0]} to {poiZoom[1]}")
    plt.xlabel("Timestamp")
    plt.ylabel("Mean |SHAP Value|")
    # Set the x ticks and labels
    x_ticks = np.arange(poiZoom[0], poiZoom[1] + xStepSize, xStepSize)
    plt.xticks(x_ticks, np.round(x_ticks, 0).astype(int))
    # y_ticks = [0, 1000, 2000, 3000, 4000]
    # plt.yticks(y_ticks, ["0", "1k", "2k", "3k", "4k"])

    plt.savefig("smaesh_syn_multiShapPlot.pdf", format='pdf')
    plt.show()

