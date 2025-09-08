# parse_VCD.py - Logan Reichling - Start 8/27/25 - UC DaSec
# Applies various filters to allow for easier VCD processing
import re
from enum import Enum


class VCDEntryDictionary:
    VCD_SIGNAL_ENTRY_PATTERN = r"^\$var ([a-z]{1,}) (\d{1,}) (.+?) (.+?) (.*?) ?\$end$"
    VCD_VALUE_CHANGE_ENTRY_PATTERN = r"^(b.+? |\d|x|z)(.+)$"
    VCD_VALUE_CHANGE_ENTRY_NOFLOAT_PATTERN = r"^(b[01]+? |\d)(.+)$"
    VCD_SIM_DELAY_ENTRY_PATTERN = r"^(#\d+)$"

    class VCDType(Enum):
        Wire = "wire"
        Reg = "reg"

    class VCDEntry:
        vcdType = None
        width = None
        identifier = None
        name = None
        depth = None

        def __init__(self, entryType, width:int, identifier:str, name:str, depth:str):
            self.vcdType = entryType
            self.width = width
            self.identifier = identifier
            self.name = name
            self.depth = depth

    # Fields
    entries = None # list()  # of VCDEntry objects
    identifierDictionary = None # dict()  # {"id":"name", ...}
    identifierTypeDictionary = None # dict()  # {"id":"type", ...}
    compiledVCDEntryPattern = None
    compiledVCDValuePattern = None

    def __init__(self):
        self.entries = list()
        self.identifierDictionary = dict()
        self.identifierTypeDictionary = dict()
        self.compiledVCDEntryPattern        = re.compile(self.VCD_SIGNAL_ENTRY_PATTERN)
        self.compiledVCDValuePattern        = re.compile(self.VCD_VALUE_CHANGE_ENTRY_PATTERN)
        self.compiledVCDDelayPattern        = re.compile(self.VCD_SIM_DELAY_ENTRY_PATTERN)
        self.compiledVCDValueNoFloatPattern = re.compile(self.VCD_VALUE_CHANGE_ENTRY_NOFLOAT_PATTERN)

    # Methods
    def addEntry(self, entryType:str, width:int, identifier:str, name:str, depth:str):
        assert type(entryType) == str, "Type of entryType is not str."
        if entryType.lower() == "wire":
            newEntryType = VCDEntryDictionary.VCDType.Wire
        elif entryType.lower() == "reg":
            newEntryType = VCDEntryDictionary.VCDType.Reg
        else:
            raise ValueError("entryType not 'wire' or 'reg'!")
        assert type(width) == int, "Type of width is not int"
        assert type(identifier) == str, "Type of identifier is not str"
        assert type(name) == str, "Type of name is not str"
        assert type(depth) == str, "Type of depth is not str"
        self.entries.append(VCDEntryDictionary.VCDEntry(newEntryType, width, identifier, name, depth))
        self.identifierDictionary[identifier] = name  # Identifier is assumed unique while the name is usually not
        self.identifierTypeDictionary[identifier] = newEntryType
        return None

if __name__ == "__main__":
    # Parameters
    includeUnknownAndFloatingValues = False
    includeDelayValues              = True
    includeSingleBitZeroValues      = False
    stopAtTimestamp = 225  # None or a number corresponding to the HW timestamp
    vcdFilePath = r"C:\Users\Logan Reichling\Desktop\smaesh_plaintext1.vcd"

    # Start code
    vcdReader1 = VCDEntryDictionary()
    with open(vcdFilePath, "r") as inFile:
        for line in inFile.readlines():
            maybeMatch = re.match(vcdReader1.compiledVCDEntryPattern, line)
            if maybeMatch is not None:
                cGroups = maybeMatch.groups()
                try:
                    vcdReader1.addEntry(cGroups[0], int(cGroups[1]), cGroups[2], cGroups[3], cGroups[4])
                except ValueError:
                    print(line)
                    print(cGroups)
                    exit(1)

    selectedVCDLinesPath = r"C:\Users\Logan Reichling\Desktop\smaesh_plaintext1.vcd"  # From same VCD as above
    savedTranslatedVCDLinesPath = r"C:\Users\Logan Reichling\Desktop\smaesh_plaintext1_translated.vcd"
    with open(selectedVCDLinesPath, "r") as inFile2:
        linesToTranslate = inFile2.readlines()
    finishedLines = list()
    counter = 0
    for i, line in enumerate(linesToTranslate):
        if includeUnknownAndFloatingValues:
            potentialMatch = re.match(vcdReader1.compiledVCDValuePattern, line)
        else:
            potentialMatch = re.match(vcdReader1.compiledVCDValueNoFloatPattern, line)
        if potentialMatch is not None:
            binaryValue, identifier = potentialMatch.groups()
            binaryValue = binaryValue.strip()
            if binaryValue in ["b0", "0"] and not includeSingleBitZeroValues:
                continue
            translatedLine = f"Line {i+1}: {vcdReader1.identifierTypeDictionary[identifier].value} {vcdReader1.identifierDictionary[identifier]} {identifier} {binaryValue}"
            finishedLines.append(translatedLine)
        else:  # potentialMatch is None:
            potentialDelayMatch = re.match(vcdReader1.compiledVCDDelayPattern, line)
            if potentialDelayMatch is not None:
                if includeDelayValues:
                    delayValue = potentialDelayMatch.groups()[0]
                    translatedLine = f"Line {i+1}: {delayValue} - HW_TS{counter}"
                    finishedLines.append(translatedLine)
                if stopAtTimestamp is not None:
                    if counter == stopAtTimestamp:
                        break
                counter += 1

    with open(savedTranslatedVCDLinesPath, "w") as outFile:
        for line in finishedLines:
            outFile.write(f"{line}\n")
