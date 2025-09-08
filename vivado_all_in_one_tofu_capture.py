# vivado_all_in_one_tofu_capture.py - Logan Reichling - Start 2/15/25 - UC DaSec
# Ver_9_5_25: Fixed batching
# Combines TOFU pipeline steps all into one mega script. Writes and uses temporary scripts when necessary
# "pip install numpy h5py tqdm matplotlib reportlab" <-- before using

# *****************************************************************
# ASSUMPTIONS:
#   * Vivado is already installed
#   * Simulation TYPE (RTL, POST_SYNTHESIS, or POST_IMPLEMENTATION, TIMING OR FUNCTIONAL)
#       o MUST BE RUN ONCE in Vivado to create the VIVADO_PROJECT_SIM_PATH
#   * Within tb_aes.v (or other file), the $dumpfile is just a file name, e.g. $dumpfile("test.vcd")

# *****************************************************************
# ----------------------- Script parameters -----------------------
# ---------------------- Change before using! ---------------------
# *****************************************************************

SAVED_TRACE_BASE_DIRECTORY = r"/home/ubuntu/Desktop/secworks"  # Final folder will be here
VIVADO_DIRECTORY           = r"/home/ubuntu/Xilinx/Vivado/2023.2"
VIVADO_PROJECT_DIRECTORY   = r"/home/ubuntu/secworks"
VIVADO_PROJECT_XPR_PATH    = r"/home/ubuntu/secworks/secworks.xpr"
VIVADO_PROJECT_SIM_BASE    = r"/home/ubuntu/secworks/secworks.sim/sim_1"
SIMULATION_MODE            = "RTL"  # RTL, POST_SYNTHESIS, or POST_IMPLEMENTATION
SIMULATION_TYPE            = "TIMING"  # TIMING or FUNCTIONAL, only matters if SIMULATION_MODE is not RTL
PLAINTEXT_FILE_PATH        = r"/home/ubuntu/50000_plaintexts.txt"
TOFU_DIRECTORY             = r"/home/ubuntu/tofu-master"
TOFU_MODE                  = r"HammingDistance"  # HammingDistance or HammingWeight
FIXED_KEY                  = [210, 213, 1, 104, 130, 131, 145, 67, 150, 158, 233, 162, 83, 167, 82, 225]
TRACES_TO_COLLECT          = 10000
BATCH_SIZE                 = 1000

# *****************************************************************
# **************** Calculated Script Parameters *******************
VIVADO_PROJECT_SIM_PATH    = VIVADO_PROJECT_SIM_BASE # Sim must be run once in Vivado; this path changes between sim modes
if SIMULATION_MODE == "RTL":
    VIVADO_PROJECT_SIM_PATH += r"/behav/xsim"
elif SIMULATION_MODE == "POST_SYNTHESIS":
    if SIMULATION_TYPE == "TIMING":
        VIVADO_PROJECT_SIM_PATH += r"/synth/timing/xsim"
    else:
        VIVADO_PROJECT_SIM_PATH += r"/synth/func/xsim"
elif SIMULATION_MODE == "POST_IMPLEMENTATION":
    if SIMULATION_TYPE == "TIMING":
        VIVADO_PROJECT_SIM_PATH += r"/impl/timing/xsim"
    else:
        VIVADO_PROJECT_SIM_PATH += r"/impl/func/xsim"
print(f"[INFO] -- Vivado sim path: {VIVADO_PROJECT_SIM_PATH}")
if TRACES_TO_COLLECT < BATCH_SIZE:
    print(f"[WARN] -- BATCH_SIZE corrected to {TRACES_TO_COLLECT} (BS: {BATCH_SIZE} > TTC: {TRACES_TO_COLLECT})")
    BATCH_SIZE = TRACES_TO_COLLECT

# *****************************************************************

import os
import re
import time
import h5py
import zipfile
import hashlib
import subprocess
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def getVCDNumber(vcdFilePath):
    pattern = re.compile(r"^.+?(?P<vcdNum>\d+)\.vcd$")
    return int(pattern.match(vcdFilePath).group('vcdNum'))


def generateStartScript(startScriptPath, tclScriptName):
    scriptContents = [
        '#!/bin/bash',
        f'VIVADO_PATH="{VIVADO_DIRECTORY}"',
        'source "$VIVADO_PATH/settings64.sh"',
        f'vivado -mode tcl -source "{tclScriptName}"'
    ]
    with open(startScriptPath, 'w') as script:
        for line in scriptContents:
            script.write(f"{line}\n")


def generateTCLScript(tempTCLScriptName, startNum, endNum, simMode, simType):
    """
    Generates the TCL script to control the execution of Vivado as well as set internal variables
    :param tempTCLScriptName: File path to save the TCL script
    :param startNum: Starting plaintext number from the provided plaintext document
    :param endNum: Ending plaintext number from the provided plaintext document
    :param simMode: Controls the Vivado simulation mode by specifying the correct launch command
    :param simType: Controls the type of simulation ran for Post-synthesis and Post-implementation modes
    :return: None
    """
    launchCommand = None
    openRunCommand = None
    if simMode == "RTL":
        launchCommand = "launch_simulation"
    elif simMode == "POST_SYNTHESIS":
        openRunCommand = "open_run synth_1"
        launchCommand = f"launch_simulation -mode post-synthesis -type {str.lower(simType)}"
    elif simMode == "POST_IMPLEMENTATION":
        openRunCommand = "open_run impl_1"
        launchCommand = f"launch_simulation -mode post-implementation -type {str.lower(simType)}"

    scriptContents = [
        f'open_project {VIVADO_PROJECT_XPR_PATH}',
        openRunCommand,
        f'{launchCommand}',
        f'set plaintexts "{PLAINTEXT_FILE_PATH}"',
        f'set config_file "{os.path.join(VIVADO_PROJECT_SIM_PATH, "config.txt")}"',
        'set fh [open $plaintexts r]',
        r'set plaintexts [split [read $fh] "\n"]',
        'close $fh',
        'for {set i '+str(startNum)+'} {$i < '+str(endNum)+'} {incr i} {',  # each plaintext $plaintexts {
        '	set cfg_fh [open $config_file w]',
        '   set plaintext [lindex $plaintexts [expr $i - 1]]',
        '	puts $cfg_fh [format "%s,%d" $plaintext $i]',
        '	close $cfg_fh',
        '	restart',
        '	run all',
        '}',
        'exit'
    ]
    with open(tempTCLScriptName, 'w') as script:
        for line in scriptContents:
            if line is not None:
                script.write(f"{line}\n")


def generateTOFUSettings(tofuSettingsFilePath, traceName):
    jsonContents = [  # These can't be full filepaths because TOFU was coded poorly
        '{',
        f'    "vcdGlob": "{traceName}.vcd",',
        f'    "pickleGlob": "{traceName}.pickle",',
        '    "signalsFileNameLiterals": "signals_name.json",',
        '    "signalsFileName": "signals.json",',
        f'    "signalPropertiesFile": "signal_properties.pickle",',
        f'    "leakageModel": "{TOFU_MODE}",',
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


def removeEmptyLinesOneFile(filepath):
    with open(filepath, "r+") as vcd:
        lines = vcd.readlines()
        vcd.seek(0)
        vcd.truncate(0)
        for line in lines:
            if line.strip():
                vcd.write(line)


def ensureParametersCorrect():
    global SAVED_TRACE_BASE_DIRECTORY, VIVADO_DIRECTORY, VIVADO_PROJECT_DIRECTORY, VIVADO_PROJECT_XPR_PATH, \
        VIVADO_PROJECT_SIM_PATH, PLAINTEXT_FILE_PATH, TOFU_DIRECTORY, TRACES_TO_COLLECT, SIMULATION_MODE
    listOfParameters = [SAVED_TRACE_BASE_DIRECTORY, VIVADO_DIRECTORY, VIVADO_PROJECT_DIRECTORY, VIVADO_PROJECT_XPR_PATH,
                        VIVADO_PROJECT_SIM_PATH, PLAINTEXT_FILE_PATH, TOFU_DIRECTORY]
    exitFlag = False
    for parameter in listOfParameters:
        if not os.path.exists(parameter):
            print(f"Parameter path: {parameter} not found!")
            exitFlag = True
    with open(PLAINTEXT_FILE_PATH, 'r') as hexPlaintexts:
        numberOfPlaintexts = len(hexPlaintexts.readlines())
        if numberOfPlaintexts < TRACES_TO_COLLECT:
            print(f"Not enough plaintexts for specified number of traces! {numberOfPlaintexts} < {TRACES_TO_COLLECT}")
    if SIMULATION_MODE not in ['RTL', 'POST_SYNTHESIS', 'POST_IMPLEMENTATION']:
        print(f"Simulation type not specified or incorrect, should be RTL, POST_SYNTHESIS, or POST_IMPLEMENTATION.")
        exitFlag = True
    if SIMULATION_MODE != "RTL":
        if SIMULATION_TYPE == "" or SIMULATION_TYPE not in ['TIMING', 'FUNCTIONAL']:
            print(f"Simulation type not specified or incorrect, should be TIMING or FUNCTIONAL.")
            exitFlag = True
    if exitFlag:
        print("One or more necessary parameters not found! Exiting...")
        exit(1)


def loadData(datasetPath):
    createdDataset = np.load(datasetPath)
    try:
        loadedTraces, loadedPlaintext, loadedKey = createdDataset['power_trace'], createdDataset['plain_text'], createdDataset['key']
    except KeyError:
        print("ERROR: Dataset retrieval with dict keys 'power_trace', 'plain_text', and 'key' unsuccessful!")
        loadedTraces, loadedPlaintext, loadedKey = None, None, None
    return loadedTraces, loadedPlaintext, loadedKey


def exportReproducibilityStats(outputPath, startingTime, endingTime, totalTiming, totalVCDTiming, totalEmptyLineTiming,
                               totalTOFUTiming, totalCombineTiming, intermediaryFileDir, generatedScriptsDir, datasetKey,
                               plaintextLen, startShFilePath, tclPath, jsonFilePath, outputTraceFilePath,
                               decimalPlaintexts):
    global SAVED_TRACE_BASE_DIRECTORY, VIVADO_DIRECTORY, VIVADO_PROJECT_DIRECTORY, VIVADO_PROJECT_XPR_PATH, \
        VIVADO_PROJECT_SIM_PATH, PLAINTEXT_FILE_PATH, TOFU_DIRECTORY

    _, datasetName = os.path.split(outputTraceFilePath+".npz")

    t, p, k = loadData(outputTraceFilePath+".npz")
    successfulCreationTest = True
    plaintextOrderingTest = True
    if t is None:
        successfulCreationTest = False
        plaintextOrderingTest = False
    else:
        for i2 in range(len(decimalPlaintexts)):
            for value in range(16):
                if p[i2][value] != decimalPlaintexts[i2][value]:
                    print(f"ERROR: Plaintext mismatch at i = {i2}:{value}")
                    plaintextOrderingTest = False

    reprodLog = list()
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation started on {startingTime.strftime('%m-%d-%Y_%H:%M:%S')} and finished on {endingTime.strftime('%m-%d-%Y_%H:%M:%S')}.")
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation timings: ")
    reprodLog.append(f"               Total time: {totalTiming:.2f} seconds")
    reprodLog.append(f"      VCD Generation time: {totalVCDTiming:.2f} seconds")
    reprodLog.append(f"  Empty Line Removal time: {totalEmptyLineTiming:.2f} seconds")
    reprodLog.append(f"     TOFU Generation time: {totalTOFUTiming:.2f} seconds")
    reprodLog.append(f"    Combine H5 files time: {totalCombineTiming:.2f} seconds")
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation input parameters:")
    reprodLog.append(f"  SAVED_TRACE_BASE_DIRECTORY: {SAVED_TRACE_BASE_DIRECTORY}")
    reprodLog.append(f"            VIVADO_DIRECTORY: {VIVADO_DIRECTORY}")
    reprodLog.append(f"    VIVADO_PROJECT_DIRECTORY: {VIVADO_PROJECT_DIRECTORY}")
    reprodLog.append(f"     VIVADO_PROJECT_XPR_PATH: {VIVADO_PROJECT_XPR_PATH}")
    reprodLog.append(f"     VIVADO_PROJECT_SIM_PATH: {VIVADO_PROJECT_SIM_PATH}")
    reprodLog.append(f"         PLAINTEXT_FILE_PATH: {PLAINTEXT_FILE_PATH}")
    reprodLog.append(f"              TOFU_DIRECTORY: {TOFU_DIRECTORY}")
    reprodLog.append(f"                   TOFU_MODE: {TOFU_MODE}")
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation dataset parameters:")
    reprodLog.append(f"            Dataset Key: {datasetKey}")
    reprodLog.append(f"           Dataset Name: {datasetName}")
    reprodLog.append(f"         Dataset Length: {plaintextLen}")
    reprodLog.append(f"  Dataset Hash (SHA256): {hashlib.sha256(open(outputTraceFilePath+'.npz', 'rb').read()).hexdigest()}")
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation created files and folders:")
    reprodLog.append(f"  Vivado Project Sim Directory: {VIVADO_PROJECT_SIM_PATH}")
    reprodLog.append(f"          *** {plaintextLen} VCDs")
    reprodLog.append(f"  Output Directory: {outputPath}")
    reprodLog.append(f"          Intermediate Files Directory: {intermediaryFileDir}")
    reprodLog.append(f"                          *** {plaintextLen}x2 pickle and H5 files")
    reprodLog.append(f"                          *** {jsonFilePath}")
    reprodLog.append(f"          Generation Scripts Directory: {generatedScriptsDir}")
    reprodLog.append(f"                          *** {startShFilePath}")
    reprodLog.append(f"                          *** {tclPath}")
    reprodLog.append(f"          Output Trace File: {outputTraceFilePath+'.npz'}")
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation tests:")
    reprodLog.append(f"  Successful Creation Test: {'Pass' if successfulCreationTest else 'FAIL <------ !!!'}")
    reprodLog.append(f"   Plaintext Ordering Test: {'Pass' if plaintextOrderingTest else 'FAIL <------ !!!'}")

    reprodOutputLogFile = os.path.join(outputPath, f"reproducibility_log_{datetime.today().strftime('%m_%d_%Y__%H_%M')}.txt")
    # Write each line to the log file
    with open(reprodOutputLogFile, 'w') as f:
        for line in reprodLog:
            f.write(f"{line}\n")

    plt.clf()
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized Voltage Drop")
    plt.plot(t[0], color='r')
    plt.savefig(os.path.join(outputPath, "traceFigure.png"))

    # Also create a PDF of the report
    def setPDFMetadata(canvas, doc):
        canvas.setTitle("Reproducibility Report")
        canvas.setSubject("Side-channel Analysis Dataset Documentation")

    pdf_path = os.path.join(outputPath, "reproducibility_report.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch
    )

    # Custom styles
    styles = getSampleStyleSheet()
    style_normal = styles['Normal']
    style_normal.fontName = 'Times-Roman'
    style_normal.fontSize = 12
    style_normal.leading = 14

    style_header = ParagraphStyle(
        'Header',
        parent=style_normal,
        fontSize=14,
        spaceAfter=20,
        alignment=1  # Center alignment
    )

    # Add log content to elements
    elements = []
    for line in reprodLog:
        elements.append(Paragraph(line.replace(" * ", "&nbsp;&nbsp;* "), style_normal))

    # elements.append(PageBreak())  # Put image on a new page
    image_path = os.path.join(outputPath, "traceFigure.png")
    elements.append(Paragraph(f"Example Trace from {datasetName} dataset:", style_header))

    # Load and scale image
    img = Image(image_path)
    img_width = 6 * inch  # Set desired width
    scaling_factor = img_width / img.drawWidth
    img.drawWidth = img_width
    img.drawHeight = img.drawHeight * scaling_factor
    img_table = Table([[img]], colWidths=doc.width)
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    elements.append(img_table)
    doc.build(elements, onFirstPage=setPDFMetadata)  # Build PDF and export


# Main
if __name__ == '__main__':
    startTraceCollectionTime = time.time()

    ### 0. Check to see if all parameters exist and make output directories
    ensureParametersCorrect()
    startTime = datetime.now()
    print(f"Starting synthetic trace creation at {startTime}")
    outputDir = os.path.join(SAVED_TRACE_BASE_DIRECTORY, f"Trace_Collection_{datetime.today().strftime('%m_%d_%Y__%H_%M')}")
    intermediateFiles = os.path.join(outputDir, "Intermediary_Files")
    generationScripts = os.path.join(outputDir, "Generation_Scripts")
    os.makedirs(outputDir, exist_ok=True)
    os.makedirs(intermediateFiles, exist_ok=True)
    os.makedirs(generationScripts, exist_ok=True)

    ### 1. Make VCD Files using Vivado
    startVCDGenerationTime = time.time()
    tclFilePath = os.path.join(generationScripts, "launch_simulation.tcl")
    generateStartScript(os.path.join(generationScripts, "run_vivado.sh"), tclFilePath)
    for item in os.listdir(VIVADO_PROJECT_SIM_PATH):  # Remove leftovers
        itemPath = os.path.join(VIVADO_PROJECT_SIM_PATH, item)
        if os.path.isfile(itemPath):
            if item in ["config.txt"]:
                subprocess.run([f'rm -f {itemPath}'], shell=True)
            if itemPath[-4:] == ".vcd":
                subprocess.run([f'rm -f {itemPath}'], shell=True)

    # Batch Vivado to avoid memory leaks,
    batchesOfTraces     = TRACES_TO_COLLECT // BATCH_SIZE
    remainingTraceBatch = TRACES_TO_COLLECT - (batchesOfTraces * BATCH_SIZE)
    subprocess.run([f'touch {os.path.join(VIVADO_PROJECT_DIRECTORY, "config.txt")}'], shell=True)
    subprocess.run([f'chmod +x {os.path.join(generationScripts, "run_vivado.sh")}'], shell=True)
    for i in tqdm(range(batchesOfTraces)):
        print(f"Batch: {(i * BATCH_SIZE + 1)} to {((i + 1) * BATCH_SIZE + 1)}")
        generateTCLScript(tclFilePath, (i * BATCH_SIZE + 1), ((i + 1) * BATCH_SIZE + 1), SIMULATION_MODE, SIMULATION_TYPE)
        time.sleep(1)
        subprocess.run([f'{os.path.join(generationScripts, "run_vivado.sh")}'], shell=True, stdout=subprocess.DEVNULL)
    if remainingTraceBatch > 0:
        print(f"Remaining Traces: {(batchesOfTraces * BATCH_SIZE + 1)} to {(batchesOfTraces * BATCH_SIZE + remainingTraceBatch + 1)}")
        generateTCLScript(tclFilePath, (batchesOfTraces * BATCH_SIZE + 1),
                          (batchesOfTraces * BATCH_SIZE + remainingTraceBatch + 1), SIMULATION_MODE, SIMULATION_TYPE)
        time.sleep(1)
        subprocess.run([f'{os.path.join(generationScripts, "run_vivado.sh")}'], shell=True, stdout=subprocess.DEVNULL)
    subprocess.run([f'rm -f {os.path.join(VIVADO_PROJECT_SIM_PATH, "plaintextx.vcd")}'], shell=True)

    # Account for any VCDs that Vivado missed (due to random segfaults, etc)
    makeupRuns = list()
    while True:  # Do while
        vcdFileList = list()
        for item in os.listdir(VIVADO_PROJECT_SIM_PATH):  # Not necessarily in order, need to sort
            itemPath = os.path.join(VIVADO_PROJECT_SIM_PATH, item)
            if os.path.isfile(itemPath):
                if itemPath[-4:] == ".vcd":
                    vcdFileList.append(itemPath)
        vcdFileList = sorted(vcdFileList, key=getVCDNumber)[:TRACES_TO_COLLECT]
        i = 1
        vcdNum = 0
        for vcd in vcdFileList:  # Assuming any makeup run isn't more than a batch
            vcdNum = getVCDNumber(vcd)
            if vcdNum != i:
                makeupRuns.append([i, vcdNum])
                runSize = vcdNum - i
                i += (runSize + 1)
            else:
                i += 1
        if vcdNum != TRACES_TO_COLLECT:
            makeupRuns.append([vcdNum + 1, TRACES_TO_COLLECT + 1])
        for makeupRun in makeupRuns:
            generateTCLScript(tclFilePath, makeupRun[0], makeupRun[1], SIMULATION_MODE, SIMULATION_TYPE)
            time.sleep(1)
            subprocess.run([f'{os.path.join(generationScripts, "run_vivado.sh")}'], shell=True, stdout=subprocess.DEVNULL)
        if len(makeupRuns) == 0:
            break

    endVCDGenerationTime = time.time()
    totalVCDGenerationTime = endVCDGenerationTime - startVCDGenerationTime
    print(f"VCD Generation took {totalVCDGenerationTime:.2f} seconds")

    ### 2. Remove empty lines from all generated VCDs
    startEmptyLineTime = time.time()
    vcdFileList = list()
    for item in os.listdir(VIVADO_PROJECT_SIM_PATH):
        itemPath = os.path.join(VIVADO_PROJECT_SIM_PATH, item)
        if os.path.isfile(itemPath):
            if itemPath[-4:] == ".vcd":
                vcdFileList.append(itemPath)
                removeEmptyLinesOneFile(itemPath)
    vcdFileList = sorted(vcdFileList, key=getVCDNumber)
    for i in range(len(vcdFileList)):  # Cannot use wildcard move
        subprocess.run([f'mv {vcdFileList[i]} {intermediateFiles}'], shell=True)
        vcdFileList[i] = os.path.join(intermediateFiles, os.path.split(vcdFileList[i])[1])
    endEmptyLineTime = time.time()
    totalEmptyLineTime = endEmptyLineTime - startEmptyLineTime
    print(f"Empty Line Removal took {totalEmptyLineTime:.2f} seconds")

    ### 3. Run TOFU to generate many h5 files
    startTOFUTime = time.time()
    currentSettingsFile = None
    for vcdFile in tqdm(vcdFileList):
        _, plaintextFileName = os.path.split(vcdFile)
        currentSettingsFile = os.path.join(intermediateFiles, "settings_example.json")
        generateTOFUSettings(currentSettingsFile, plaintextFileName[:-4])
        subprocess.run([f'python3 "{os.path.join(TOFU_DIRECTORY, "parse.py")}" --settings "{currentSettingsFile}" >/dev/null'], shell=True)
        subprocess.run([f'python3 "{os.path.join(TOFU_DIRECTORY, "synthesize.py")}" --settings "{currentSettingsFile}" >/dev/null'], shell=True)
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
        for plaintext in plaintexts[:TRACES_TO_COLLECT]:
            decimalPlaintext = list()
            strippedPlaintext = plaintext.strip()
            for i in range(0, len(strippedPlaintext), 2):
                decimalByte = int(strippedPlaintext[i:i+2], 16)
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
    outputTraceFile = os.path.join(outputDir, f"Synthetic{TOFU_MODE}_K1_{int((len(traces)/1000)+0.5)}k")
    np.savez(outputTraceFile, power_trace=power_trace, plain_text=plain_text, key=key)
    endCombineTime = time.time()
    totalCombineTime = endCombineTime - startCombineTime
    print(f"Combining H5 files took {totalCombineTime:.2f} seconds")

    # Compress intermediate files to save space
    with zipfile.ZipFile(os.path.join(outputDir, "Intermediate_Files.zip"), "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zipFile:
        for item in os.listdir(intermediateFiles):
            itemPath = os.path.join(intermediateFiles, item)
            if os.path.isfile(itemPath):
                zipFile.write(itemPath)
    subprocess.run([f'rm -rf {intermediateFiles}'], shell=True)

    ### 6. Done! Export metadata and exit
    endTraceCollectionTime = time.time()
    endTime = datetime.now()
    traceCollectionTime = endTraceCollectionTime - startTraceCollectionTime
    exportReproducibilityStats(outputDir, startTime, endTime, traceCollectionTime, totalVCDGenerationTime,
                               totalEmptyLineTime, totalTOFUTime, totalCombineTime, intermediateFiles,
                               generationScripts, key, TRACES_TO_COLLECT,
                               os.path.join(generationScripts, "run_vivado.sh"), tclFilePath, currentSettingsFile,
                               outputTraceFile, decimalPlaintexts)
    print(f"Trace collection took {traceCollectionTime:.2f} seconds")
    print(f"Trace collection finished at {datetime.now()}")
    print("All done! Exiting...")


