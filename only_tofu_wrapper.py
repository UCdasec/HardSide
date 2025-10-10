import os
import re
import subprocess
import time
from datetime import datetime

import h5py
import numpy as np
from tqdm import tqdm


def getVCDNumber(vcdFilePath):
    pattern = re.compile(r"^.+?(?P<vcdNum>\d+)\.vcd$")
    return int(pattern.match(vcdFilePath).group('vcdNum'))


def returnSortedVCDPathsFromDir(dirPath):
    vcdFilePs = list()
    for vcdF in os.listdir(dirPath):  # Not necessarily in order, need to sort
        itemP = os.path.join(dirPath, vcdF)
        if os.path.isfile(itemP):
            if itemP[-4:] == ".vcd":
                vcdFilePs.append(itemP)
    vcdFilePs = sorted(vcdFilePs, key=getVCDNumber)
    return vcdFilePs


def generateTOFUSettings(tofuSettingsFilePath, traceName, tofuMode):
    jsonContents = [
        '{',
        f'    "vcdGlob": "{traceName}.vcd",',
        f'    "pickleGlob": "{traceName}.pickle",',
        '    "signalsFileNameLiterals": "signals_name.json",',
        '    "signalsFileName": "signals.json",',
        f'    "signalPropertiesFile": "signal_properties.pickle",',
        f'    "leakageModel": "{tofuMode}",',
        '    "window": false,',
        '    "windowFrom": null,',
        '    "windowTo": null,',
        '    "valueExtractFunction": "valueExtractIndex",',
        '    "writeTraces": true,',
        '    "writeTracesBatchSize": 10,',
        f'    "traceFileName": "{traceName}.h5",',
        '    "align": false,',
        '    "downsample": 1e5,',
        '    "format": "lascar"',
        '}'
    ]
    with open(tofuSettingsFilePath, 'w') as jsonFile:
        for line in jsonContents:
            jsonFile.write(f"{line}\n")


if __name__ == "__main__":
    # Parameters
    TOFU_MODE = "HammingWeight"
    TOFU_DIRECTORY = r""
    PLAINTEXT_FILE_PATH = r""  # Make sure it is LE for SMAesH
    FIXED_KEY = r""
    COLLECTED_VCDS_DIR = r""
    BASE_DIR = r""


    # Get path for each VCD in directory
    startTraceCollectionTime = time.time()
    startTOFUTime = time.time()
    vcdFileList = returnSortedVCDPathsFromDir(COLLECTED_VCDS_DIR)
    ### 3. Run TOFU to generate many h5 files
    currentSettingsFile = None
    for vcdFile in tqdm(vcdFileList):
        _, plaintextFileName = os.path.split(vcdFile)
        currentSettingsFile = os.path.join(COLLECTED_VCDS_DIR, "settings_example.json")
        generateTOFUSettings(currentSettingsFile, plaintextFileName[:-4], TOFU_MODE)
        subprocess.run(
            ['python3', f'{os.path.join(TOFU_DIRECTORY, "parse.py")}', '--settings', f'{currentSettingsFile}'],
            shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(
            ['python3', f'{os.path.join(TOFU_DIRECTORY, "synthesize.py")}', '--settings', f'{currentSettingsFile}'],
            shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    endTOFUTime = time.time()
    totalTOFUTime = endTOFUTime - startTOFUTime
    print(f"TOFU Generation took {totalTOFUTime:.2f} seconds")

    ### 4. Combine all H5 files into the final npz dataset
    startCombineTime = time.time()
    key = np.array(FIXED_KEY, dtype=np.uint8)

    # Collect plaintexts
    decimalPlaintexts = list()
    with open(PLAINTEXT_FILE_PATH, 'r') as hexPlaintexts:
        plaintexts = hexPlaintexts.readlines()
        for plaintext in plaintexts[:len(vcdFileList)]:
            decimalPlaintext = list()
            strippedPlaintext = plaintext.strip()
            for i in range(0, len(strippedPlaintext), 2):
                decimalByte = int(strippedPlaintext[i:i + 2], 16)
                decimalPlaintext.append(decimalByte)
            decimalPlaintexts.append(decimalPlaintext)
    plain_text = np.array(decimalPlaintexts, dtype=np.uint8)

    # Collect traces
    h5FilePaths = list()
    for vcdFileName in vcdFileList:
        h5FilePaths.append("".join([vcdFileName[:-4], ".h5"]))
    traces = list()
    for h5FilePath in h5FilePaths:
        with h5py.File(h5FilePath, 'r') as h5File:
            trace = h5File['leakages'][:]
            traces.append(trace)
    power_trace = np.concatenate(traces, axis=0)
    outputTraceFile = os.path.join(BASE_DIR, f"Synthetic{TOFU_MODE}_K1_{len(traces)}")
    np.savez(outputTraceFile, power_trace=power_trace, plain_text=plain_text, key=key)
    endCombineTime = time.time()
    totalCombineTime = endCombineTime - startCombineTime
    print(f"Combining H5 files took {totalCombineTime:.2f} seconds")

    ### 6. Done! Export metadata and exit
    endTraceCollectionTime = time.time()
    endTime = datetime.now()
    traceCollectionTime = endTraceCollectionTime - startTraceCollectionTime
    # exportReproducibilityStats(BASE_DIR, startTime, endTime, traceCollectionTime, totalVCDGenerationTime,
    #                            totalEmptyLineTime, totalTOFUTime, totalCombineTime, intermediateFiles,
    #                            generationScripts, key, TRACES_TO_COLLECT, vivadoScriptPath, tclFilePath,
    #                            currentSettingsFile, outputTraceFile, decimalPlaintexts)
    print(f"Trace collection took {traceCollectionTime:.2f} seconds")
    print(f"Trace collection finished at {datetime.now()}")
    print("All done! Exiting...")
