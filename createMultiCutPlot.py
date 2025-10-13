# createMultiCutPlot.py
import numpy as np
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
            return dataArray['power_trace'], dataArray['plain_text'], dataArray['key']
    except (TypeError, ValueError) as e:
        print(f"Non-NPZ file format or unrecognized NPZ dataset")
        return None


# Main
if __name__ == "__main__":
    # Parameters
    selectedTrace = 0
    poiZoom = (0, 505)
    xStepSize = 125
    yStepSize = 500
    datasets = [
        r"C:\Users\Logan Reichling\Desktop\HardSide Project\SMAesH_wo_endian_reverse_and_le_key_POST_SYNTHESIS_FUNCTIONAL_SyntheticHammingWeight_K1_50k.npz",
        r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\SMAesHHW_LittleEndian_LEKey_PostSyn_K1_50000_Cut1_new.npz",
        r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\SMAesHHW_LittleEndian_LEKey_PostSyn_K1_50000_Cut2_new.npz",
        r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\SMAesHHW_LittleEndian_LEKey_PostSyn_K1_50000_Cut3_new.npz",
        r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\SMAesHHW_LittleEndian_LEKey_PostSyn_K1_50000_Cut4_new.npz",
        r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\SMAesHHW_LittleEndian_LEKey_PostSyn_K1_50000_Cut5_new.npz",
        r"C:\Users\Logan Reichling\Desktop\Ablation Study Work\SMAesHHW_LittleEndian_LEKey_PostSyn_K1_50000_Cut6_new.npz",
    ]


    selectedTraces = list()
    for i, datasetPath in enumerate(datasets):
        traces, plaintexts, key = loadNpzFile(datasetPath)
        labels = None
        selectedTraces.append(traces[selectedTrace])
        print(f"Dataset {i}: {datasetPath}")
        print("Shape of one plaintext: ", np.shape(plaintexts[0]))
        print("Shape (key):", np.shape(key))
        print("Plaintext: ", end="")
        print(plaintexts[selectedTrace])
        print("Plaintext: ", end="")
        for pByte in plaintexts[selectedTrace]:
            print(f"{pByte:02x}", end="")
        print("")
        print("Key: ", end="")
        print(key)
        print("Key: ", end="")
        for keyByte in key:
            print(f"{keyByte:02x}", end="")
        print("")

    # Start plot...
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(4, 3))

    # plt.axhline(y=0, color='k', linestyle='dashed', linewidth=0.5)  # Line at 0

    colorList = ['r', 'b', 'g', 'orange', 'purple', 'teal', 'olivedrab']
    for i, trace in enumerate(selectedTraces):
        zoomed_data = trace[poiZoom[0]:poiZoom[1]]
        if i == 0:
            label = "Orig."
        else:
            label = f"Cut {i}"
        plt.plot(np.arange(poiZoom[0], poiZoom[1]), zoomed_data, label=label, color=colorList[i])

    plt.margins(x=0, y=0.03)  # Tight plot bounds (no starting and ending whitespace)
    plt.legend(loc="upper left", prop={'size': 11.5})

    plt.tight_layout(rect=[-0.12, 0.02, 1, 1])  # (left, bottom, right, top)

    # plt.title(f"{fileName} Trace POI Window index from {poiZoom[0]} to {poiZoom[1]}")
    plt.xlabel("Timestamp")
    # plt.ylabel("Normalized Voltage Drop")
    # Set the x ticks and labels
    x_ticks = np.arange(poiZoom[0], poiZoom[1], xStepSize)
    plt.xticks(x_ticks, np.round(x_ticks, 0).astype(int))
    y_ticks = [0, 1000, 2000, 3000, 4000]
    plt.yticks(y_ticks, ["0", "1k", "2k", "3k", "4k"])

    plt.savefig("smaesh_syn_multiCutPlot.pdf", format='pdf')
    plt.show()

