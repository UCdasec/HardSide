# genInfoPlots.py - Start 10/30/23 - Logan Reichling
# genInfoPlots.py is a script that takes a numpy array and plots the key rank curve, adding some helpful information
import json
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt
import re


def load_npz_file(file_name):
    """
    load_npz_file takes a file name and loads the numpy array from the file.
    :param file_name: The name of the file to load the numpy array from.
    :return: The numpy array loaded from the file.
    """
    with np.load(file_name) as dataArray:
        return dataArray['y']


def confirmShape(array):
    """
    get_shape takes a numpy array and returns the shape of the array.
    :param array: The numpy array to get the shape of.
    :return: The shape of the numpy array.
    """
    shapeNumpy = array.shape
    assert len(shapeNumpy) == 1, "Array is not one-dimensional"
    return shapeNumpy


def findZeroIndices(array):
    """
    find_zero_indices takes a numpy array and returns the indices of the first and last zero values.
    :param array: The numpy array to find the indices of the first and last zero values.
    :return: The indices of the first and last zero values as a tuple
    """
    KEY_RANK_ZERO_THRESHOLD = 0.5

    # BOUNCE_SIZE = 2
    firstBecomeZero = -1
    final_become_zero = -1
    for i in range(len(array)):
        if array[i] <= KEY_RANK_ZERO_THRESHOLD and firstBecomeZero == -1:
            firstBecomeZero = i
        # elif array[i] <= KEY_RANK_ZERO_THRESHOLD and firstBecomeZero != -1:
        #    final_become_zero = i
        #    break
    return firstBecomeZero, final_become_zero


def create_cropped_array(array, becomeZero):
    """
    create_cropped_array takes a numpy array and returns a cropped version of the array.
    :param array: The numpy array to crop.
    :param last_become_zero: The index of the last zero value in the array.
    :return: The cropped numpy array, to better visualize the curve.
    """
    return array[:becomeZero + 20]


def plotDefaultRankCurve(array, byteStrNum, leakageModel, show=False):
    plt.figure(figsize=(8, 6))
    plt.plot(array, color='red')
    plt.title('Leakage model: {}, target byte: {}'.format(leakageModel, byteStrNum))
    plt.xlabel('Number of trace')
    plt.ylabel('Key Rank')
    if show:
        plt.show()
        plt.close()


def plotKeyRankCurve(array, byteStrNum, xLabel, yLabel, device, leakageModel, firstBecomeZero, lastBecomeZero,
                     show=False):
    """
    plot_key_rank_curve takes a numpy array and plots the key rank curve.
    :param leakageModel:
    :param show:
    :param yLabel:
    :param device:
    :param byteStrNum:
    :param xLabel:
    :param array: The numpy array to plot the key rank curve of.
    :param firstBecomeZero: The index of the first zero value in the array.
    :param lastBecomeZero: The index of the last zero value in the array.
    :return: None
    """
    plt.plot(array, label="Key Rank Curve")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(f"{device} Key Rank Curve for Byte {byteStrNum} Target.\nLeakage Model: {leakageModel}")
    if firstBecomeZero != -1:
        plt.axvline(x=firstBecomeZero, color='g', linestyle="--", label=f"First zero at {firstBecomeZero}")
    if lastBecomeZero != -1:
        plt.axvline(x=lastBecomeZero, color='r', linestyle="--", label=f"Last zero at {lastBecomeZero}")
    plt.legend()
    if show:
        plt.show()


def createTimeGraph(array, xLabel, yLabel, roundNum, datasetName, leakageModel, exportDir, saveName, show=False):
    """
    plotRoundTimes takes and plots an array representing the time per pruning round.
    :param array:           Array of time per round data
    :param xLabel:          Label for x-axis, i.e. 'Round'
    :param yLabel:          Label for y-axis, i.e. 'Time (s)'
    :param roundNum:        Number of rounds
    :param datasetName:     Name of the dataset
    :param leakageModel:    Leakage model utilized
    :param exportDir:       Directory to save within
    :param saveName:        Time graph save name
    :param show:            Boolean to trigger the graph to pop-up; not for headless systems
    :return: None
    """
    plt.bar(x=range(len(array)), height=array, label="Time per round", color='#11DD11')

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(
        f"Iterative pruning time per round over {roundNum} rounds for {datasetName}.\nLeakage Model: {leakageModel}")
    for i in range(len(array)):
        plt.text(i, array[i], f"   {array[i]:.1f} s", ha='center', va='baseline', rotation=90)
    plt.text(len(array), max(array) + 1400 * 0.9, f"Total time: {sum(array):.1f} s", ha='right', va='top', rotation=0,
             bbox=dict(facecolor='none', edgecolor='#dddddd', boxstyle='round'))
    y_ticks = np.arange(0, max(array) + 20000, 10000)
    # plt.xticks(x_ticks, np.round(x_ticks, 0).astype(int))
    plt.yticks(y_ticks)

    plt.subplots_adjust(left=0.15, right=0.875, top=0.8, bottom=0.125)
    plt.legend()
    if show:
        plt.show()
    plt.savefig(os.path.join(exportDir, saveName + '.png'))
    plt.savefig(os.path.join(exportDir, saveName + '.pdf'))


# Can run as console application or run directly from IDE
# Console application: python genInfoPlots.py <path_to_npz_file.npz> <path_to_save_figure.png>
if __name__ == "__main__":
    test = np.array(json.loads(
        "[4139.13254737854, 499.13811111450195, 97.53126573562622, 202.5841748714447, 682.4539387226105, 185.3782615661621, 637.9889011383057, 4687.191566944122, 1023.6001524925232, 609.9150047302246, 1244.1994426250458]"))
    createTimeGraph(test, "Round", "Time (s)", len(test), "TEST", "ID",
                    "C:\\Users\\Logan Reichling\\PycharmProjects\\IterativePruning\\misc", "test", True)

    # if not sys.stdout.isatty():
    #     # Load single numpy .npz file
    #     data = load_npz_file(
    #         "testDir/CNN_B1_XMegaUnmasked_1800to2800_HW_test_10000t_X1K1200k/rank_dir/ranking_raw_data.npz")
    #
    #     # Save the shape of the numpy array to a variable and confirm that it is a one-dimensional array
    #     shape = confirmShape(data)
    #     first_become_zero, last_become_zero = findZeroIndices(data)  # Find important indices
    #     if last_become_zero != -1:
    #         cropped_data = create_cropped_array(data, last_become_zero)  # Crop original dataset
    #     elif first_become_zero != -1:
    #         cropped_data = create_cropped_array(data, first_become_zero)
    #     else:
    #         cropped_data = data
    #     plotKeyRankCurve(cropped_data, "TEST", "Number of Traces", "Key Rank", "TEST", "TEST", first_become_zero,
    #                      last_become_zero, show=True)  # Plot the rank curve
    #     plt.savefig("key_rank_curve_test_cropped.png")
    #
    # else:
    #     passedArgs = sys.argv
    #     if len(passedArgs) == 2:
    #         if os.path.exists(passedArgs[1]) and os.path.isfile(passedArgs[1]):
    #             data = load_npz_file(passedArgs[1])
    #             shape = confirmShape(data)
    #             first_become_zero, last_become_zero = findZeroIndices(data)
    #             if last_become_zero != -1:
    #                 cropped_data = create_cropped_array(data, last_become_zero)  # Crop original dataset
    #             elif first_become_zero != -1:
    #                 cropped_data = create_cropped_array(data, first_become_zero)
    #             else:
    #                 cropped_data = data
    #             # Plot the rank curve
    #             plotKeyRankCurve(cropped_data, "TEST", "Number of Traces", "Key Rank", "TEST", "TEST",
    #                              first_become_zero, last_become_zero, show=True)
    #             plt.savefig("key_rank_curve_test_cropped.png")
    #
    #         # Batch processing for all tests, should be the preferred method
    #         # If passed a dir, process all directories that have correct format
    #         # Ex: command "genInfoPlots.py /SoftPower-master/", will process in all like:
    #         # "/SoftPower-master/CNN_B0_XMegaUnmasked_1800to2800_HW_test_10000t_X1K1100k/rank_dir/ranking_raw_data.npz"
    #         # TODO will produce a report in the top directory passed, e.g. "/SoftPower-master" with all results
    #         elif os.path.exists(passedArgs[1]) and os.path.isdir(passedArgs[1]):
    #             resultFigPaths = list()
    #             resultFigDescriptions = list()
    #             dirName = os.path.basename(os.path.normpath(passedArgs[1]))
    #             namingMatch = re.compile(
    #                 r"^CNN(.*?)_B(\d{1,2})_(.+?)_(\d{1,5})to(\d{3,5})_(HW|ID)_test_(\d{3,5})t_(.+)")
    #             for fileOrFolder in os.scandir(passedArgs[1]):
    #                 if fileOrFolder.is_dir():
    #                     # Check if directory matches standard directory name
    #                     directoryMatches = namingMatch.match(fileOrFolder.name)
    #                     if directoryMatches is not None:
    #                         matchGroups = directoryMatches.groups()
    #                         # Check if directory has a ranking_raw_data.npz file
    #                         if os.path.exists(fileOrFolder.path + "/rank_dir/ranking_raw_data.npz"):
    #                             data = load_npz_file(os.path.join(fileOrFolder.path + "/rank_dir/ranking_raw_data.npz"))
    #                             shape = confirmShape(data)
    #                             first_become_zero, last_become_zero = findZeroIndices(data)
    #                             if last_become_zero != -1:
    #                                 cropped_data = create_cropped_array(data, last_become_zero)
    #                             elif first_become_zero != -1:
    #                                 cropped_data = create_cropped_array(data, first_become_zero)
    #                             else:
    #                                 cropped_data = data
    #
    #                             print(f"First 10 results from {fileOrFolder.name}")
    #                             print(cropped_data[:10])
    #
    #                             # Generate graph with filled-in info
    #                             plotKeyRankCurve(cropped_data, matchGroups[0], "Number of Traces", "Key Rank",
    #                                              matchGroups[1], matchGroups[4], first_become_zero, last_become_zero)
    #                             plt.savefig(os.path.join(fileOrFolder.path +
    #                                                      f"/rank_dir/key_rank_curve_B{matchGroups[0]}_cropped.png"))
    #                             resultFigPaths.append(os.path.join(fileOrFolder.path +
    #                                                                f"/rank_dir/key_rank_curve_B{matchGroups[0]}_cropped.png"))
    #                             resultFigDescriptions.append(f"Dataset: {matchGroups[6]}\nByte: {matchGroups[0]}")
    #                             plt.clf()
    #
    #                         else:
    #                             print(f"WARNING: Result folder \"{fileOrFolder.path}\" missing npz data!")
    #                     else:
    #                         print(f"WARNING: No naming matches on passed directory!")
    #             # Generate final report if numpy data was processed
    #             if len(resultFigPaths) > 0:
    #                 pass
    #
    #         else:
    #             print("File/directory does not exist.")
    #         print("Done!")
    #
    #     elif len(passedArgs) == 3:
    #         data = load_npz_file(passedArgs[1])
    #         shape = confirmShape(data)
    #         first_become_zero, last_become_zero = findZeroIndices(data)
    #         if last_become_zero != -1:
    #             cropped_data = create_cropped_array(data, last_become_zero)  # Crop original dataset
    #         elif first_become_zero != -1:
    #             cropped_data = create_cropped_array(data, first_become_zero)
    #         else:
    #             cropped_data = data
    #         # Plot the rank curve
    #         plotKeyRankCurve(cropped_data, "TEST", "Number of Traces", "Key Rank", "TEST", "TEST", first_become_zero,
    #                          last_become_zero, show=True)
    #         plt.savefig(passedArgs[2])
