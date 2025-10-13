import hashlib
import os
import re
import subprocess
import time
from datetime import datetime
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle
import h5py
import numpy as np
from tqdm import tqdm


def getVCDNumber(vcdFilePath):
    pattern = re.compile(r"^.+?(?P<vcdNum>\d+)\.vcd$")
    return int(pattern.match(vcdFilePath).group('vcdNum'))


def loadData(datasetPath):
    createdDataset = np.load(datasetPath)
    try:
        loadedTraces, loadedPlaintext, loadedKey = createdDataset['power_trace'], createdDataset['plain_text'], createdDataset['key']
    except KeyError:
        print("ERROR: Dataset retrieval with dict keys 'power_trace', 'plain_text', and 'key' unsuccessful!")
        loadedTraces, loadedPlaintext, loadedKey = None, None, None
    return loadedTraces, loadedPlaintext, loadedKey


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


def exportTOFUONLYReproducibilityStats(outputPath, startingTime, endingTime, totalTiming,
                               totalTOFUTiming, totalCombineTiming, intermediaryFileDir, datasetKey,
                               plaintextLen, jsonFilePath, outputTraceFilePath,
                               decimalPlaintextsForTest):
    _, datasetName = os.path.split(outputTraceFilePath+".npz")
    t, p, k = loadData(outputTraceFilePath+".npz")
    successfulCreationTest = True
    plaintextOrderingTest = True
    if t is None:
        successfulCreationTest = False
        plaintextOrderingTest = False
    else:
        for i2 in range(len(decimalPlaintextsForTest)):
            for value in range(16):
                if p[i2][value] != decimalPlaintextsForTest[i2][value]:
                    print(f"ERROR: Plaintext mismatch at i = {i2}:{value}")
                    plaintextOrderingTest = False

    reprodLog = list()
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"-----> TOFU ONLY PROCESSING OF PREVIOUSLY GENERATED VCDS  <----- ")
    reprodLog.append(f"Trace generation started on {startingTime.strftime('%m-%d-%Y_%H:%M:%S')} and finished on {endingTime.strftime('%m-%d-%Y_%H:%M:%S')}.")
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation timings: ")
    reprodLog.append(f"               Total time: {totalTiming:.2f} seconds")
    reprodLog.append(f"     TOFU Generation time: {totalTOFUTiming:.2f} seconds")
    reprodLog.append(f"    Combine H5 files time: {totalCombineTiming:.2f} seconds")
    reprodLog.append(f"############################################################################")
    reprodLog.append(f"Trace generation input parameters:")
    reprodLog.append(f"  SAVED_TRACE_BASE_DIRECTORY: {outputPath}")
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
    reprodLog.append(f"  VCD Directory: {intermediaryFileDir}")
    reprodLog.append(f"          *** {plaintextLen} VCDs")
    reprodLog.append(f"  Output Directory: {outputPath}")
    reprodLog.append(f"          Intermediate Files Directory: {intermediaryFileDir}")
    reprodLog.append(f"                          *** {plaintextLen}x2 pickle and H5 files")
    reprodLog.append(f"                          *** {jsonFilePath}")
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
if __name__ == "__main__":
    # Parameters
    TOFU_MODE = "HammingWeight"
    TOFU_DIRECTORY = r""
    PLAINTEXT_FILE_PATH = r""  # Make sure it is LE for SMAesH
    FIXED_KEY = r""
    COLLECTED_VCDS_DIR = r""
    BASE_DIR = r""


    # Get path for each VCD in directory
    startDateAndTime = datetime.now()
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
    exportTOFUONLYReproducibilityStats(BASE_DIR, startDateAndTime, endTime, traceCollectionTime, totalTOFUTime,
                               totalCombineTime, COLLECTED_VCDS_DIR, key, len(traces),
                               currentSettingsFile, outputTraceFile, decimalPlaintexts)
    print(f"TOFU processing and dataset forming took {traceCollectionTime:.2f} seconds")
    print(f"Trace collection finished at {datetime.now()}")
    print("All done! Exiting...")
