import os
import re
from tqdm import tqdm
from parse_VCD import VCDEntryDictionary


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


# Main
if __name__ == "__main__":
    # Parameters
    CUT_MODE = "exclude"  # or "exclude"; include mode leaves only specified signals in time range, exclude cuts specified signals and leaves rest
    SIGNAL_NAMES = ["ina", "inb", "out"]
    TIMESTAMP_MODIFY_RANGE = (200, 500)  # TODO, does nothing right now
    DIR_PATH_TO_VCDS = r"C:\Users\Logan Reichling\Desktop\testFolder"

    # Record path of every VCD file
    vcdFilePaths = returnSortedVCDPathsFromDir(DIR_PATH_TO_VCDS)

    # Build dictionary to later translate identifiers (ASSUME EACH VCD IS FROM THE SAME IMPLEMENTATION)
    vcdReader2 = VCDEntryDictionary()
    with open(vcdFilePaths[0], "r") as inFile:
        for line in inFile.readlines():
            maybeMatch = re.match(vcdReader2.compiledVCDEntryPattern, line)
            if maybeMatch is not None:
                cGroups = maybeMatch.groups()
                try:
                    vcdReader2.addSignalEntry(cGroups[0], int(cGroups[1]), cGroups[2], cGroups[3], cGroups[4])
                except ValueError as err:
                    print(err)
                    print(line)
                    print(cGroups)
                    exit(1)

    # Ensure specified signal names are present within vcdReader dictionary
    nameLists = list(vcdReader2.identifierDictionary.values())
    nameCheckFlag = False
    for name in SIGNAL_NAMES:
        for names in nameLists:
            if name in names:
                nameCheckFlag = True
    if not nameCheckFlag:
        print("Given signal name not found within VCD file. Please check again.")
        exit(1)

    # Leave only the selected signal names
    for vcdFile in tqdm(vcdFilePaths):
        # Read in lines and close file
        tempOutLines = list()
        with open(vcdFile, "r") as inFile:
            tempLines = inFile.readlines()
        # Craft file with included or excluded signal names, other lines left the same
        for vcdLine in tempLines:
            potentialMatch = re.match(vcdReader2.compiledVCDValuePattern, vcdLine)
            if potentialMatch is not None:  # Signal update line
                _, identifier = potentialMatch.groups()
                signalNames = vcdReader2.identifierDictionary[identifier]
                signalLine = False
                if CUT_MODE == "include":  # Do not include other signals
                    for signalName in signalNames:
                        if signalName in SIGNAL_NAMES:
                            signalLine = True
                    if signalLine:
                        tempOutLines.append(vcdLine)
                    signalLine = False

                elif CUT_MODE == "exclude":  # Exclude specified signals
                    for signalName in signalNames:
                        if signalName in SIGNAL_NAMES:  # If any of the signal names are within SIGNAL_NAMES, do not add
                            signalLine = True
                    if not signalLine:
                        tempOutLines.append(vcdLine)
                    signalLine = False

            else:  # Not a signal update line, add to tempOutLines
                tempOutLines.append(vcdLine)



        # Save file back
        outLines = "".join(tempOutLines).encode()
        with open(vcdFile, "wb") as outFile:
            outFile.write(outLines)

    print("Done cutting VCDs.")


