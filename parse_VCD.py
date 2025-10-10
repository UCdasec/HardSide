# parse_VCD.py - Logan Reichling - Start 8/27/25 - UC DaSec
# Applies various filters to allow for easier VCD processing
import os.path
import re
from collections import defaultdict
from enum import Enum


class VCDEntryDictionary:
    # Patterns
    VCD_ENTRY_PATTERN                       = r"^\$var ([a-z]{1,}) (\d{1,}) (.+?) (.+?) (.*?) ?\$end$"
    VCD_VALUE_CHANGE_ENTRY_PATTERN          = r"^(b.+? |\d|x|z)(.+)$"
    VCD_VALUE_CHANGE_ENTRY_NOFLOAT_PATTERN  = r"^(b[01]+? |\d)(.+)$"
    VCD_DELAY_ENTRY_PATTERN                 = r"^(#\d+)$"
    VCD_SCOPE_DEFINITION_PATTERN            = r"^\$scope .+? (.+?) \$end$"

    # Constant Values
    BINARY_TO_HEX = {"0000": "0", "0001": "1", "0010": "2", "0011": "3", "0100": "4", "0101": "5",  "0110": "6",
                     "0111": "7", "1000": "8", "1001": "9", "1010": "A", "1011": "B", "1100": "C", "1101": "D",
                     "1110": "E", "1111": "F"}

    class VCDType(Enum):
        Wire = "wire"
        Reg = "reg"
        Parameter = "parameter"

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
    identifierModuleDictionary = None # dict()  # {"id":"module_name", ...}
    compiledVCDEntryPattern = None
    compiledVCDValuePattern = None

    def __init__(self):
        self.entries = list()
        self.identifierDictionary = defaultdict(list)
        self.identifierTypeDictionary = defaultdict(list)
        self.identifierWidthDictionary = dict()
        self.compiledVCDEntryPattern        = re.compile(self.VCD_ENTRY_PATTERN)
        self.compiledVCDValuePattern        = re.compile(self.VCD_VALUE_CHANGE_ENTRY_PATTERN)
        self.compiledVCDDelayPattern        = re.compile(self.VCD_DELAY_ENTRY_PATTERN)
        self.compiledVCDValueNoFloatPattern = re.compile(self.VCD_VALUE_CHANGE_ENTRY_NOFLOAT_PATTERN)

    # Methods
    def addSignalEntry(self, entryType:str, width:int, identifier:str, name:str, depth:str):
        assert type(entryType) == str, "Type of entryType is not str."
        if entryType.lower() == "wire":
            newEntryType = VCDEntryDictionary.VCDType.Wire
        elif entryType.lower() == "reg":
            newEntryType = VCDEntryDictionary.VCDType.Reg
        elif entryType.lower() == "parameter":
            newEntryType = VCDEntryDictionary.VCDType.Parameter
        else:
            raise ValueError("entryType not 'wire' or 'reg'!")
        assert type(width) == int, "Type of width is not int"
        assert type(identifier) == str, "Type of identifier is not str"
        assert type(name) == str, "Type of name is not str"
        assert type(depth) == str, "Type of depth is not str"
        # self.entries.append(VCDEntryDictionary.VCDEntry(newEntryType, width, identifier, name, depth))

        if len(self.identifierDictionary[identifier]) == 0:
            self.identifierDictionary[identifier].append(name)
            self.identifierTypeDictionary[identifier].append(newEntryType)
            self.identifierWidthDictionary[identifier] = width
        else:
            if name not in self.identifierDictionary[identifier]:
                self.identifierDictionary[identifier].append(name)
                self.identifierTypeDictionary[identifier].append(newEntryType)
                self.identifierWidthDictionary[identifier] = width
        return None

# Main
if __name__ == "__main__":
    # Parameters
    includeUnknownAndFloatingValues = False
    includeDelayValuesAndTimestamps = True
    finalValueFormat                = 'hex'  # Options: hex, binary, TODO: decimal, octal,
    stopAtTimestamp                 = None  # None or a number corresponding to the HW timestamp
    timestampLineVisualFillToLength = 90
    includeZeroValues               = True
    extendBinaryValuesToWidth       = True
    includeShapValueList            = False

    vcdFilePath = r"C:\Users\Logan Reichling\Desktop\New folder\plaintext5.vcd"
    savedTranslatedVCDLinesPath = r"C:\Users\Logan Reichling\Desktop\New folder\plaintext5_translated.vcd"
    optionalShapValueList = r".\shapValueList.txt"

    # Check params
    if includeShapValueList and not (os.path.exists(optionalShapValueList) and os.path.isfile(optionalShapValueList)):
        print("SHAP value printout enabled with invalid shap value list path provided.")
        exit(1)
    if vcdFilePath == savedTranslatedVCDLinesPath:
        print("vcdFilePath and savedTranslatedVCDLinesPath is the same!")
        exit(1)

    # Start code
    # Read optional shap value list
    shapValueList = list()
    if includeShapValueList:
        with open(optionalShapValueList, "r") as inFile1:
            for line in inFile1.readlines():
                shapValueLine = line.strip()
                if shapValueLine != "":
                    shapValueList.append(shapValueLine)

    # Read vcd declaration section and build dictionary
    vcdReader1 = VCDEntryDictionary()
    with open(vcdFilePath, "r") as inFile2:
        for line in inFile2.readlines():
            maybeMatch = re.match(vcdReader1.compiledVCDEntryPattern, line)
            if maybeMatch is not None:
                cGroups = maybeMatch.groups()
                try:
                    vcdReader1.addSignalEntry(cGroups[0], int(cGroups[1]), cGroups[2], cGroups[3], cGroups[4])
                except ValueError as err:
                    print(err)
                    print(line)
                    print(cGroups)
                    exit(1)

    # Read VCD value lines and translate identifiers and print nicely
    with open(vcdFilePath, "r") as inFile3:
        linesToTranslate = inFile3.readlines()
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

            # Value modification subsection
            if binaryValue in ["b0", "0"] and not includeZeroValues:
                continue
            if binaryValue[0] == "b" and extendBinaryValuesToWidth:
                binValueFullWidth = vcdReader1.identifierWidthDictionary[identifier]
                binaryValue = binaryValue[1:]
                binaryValue = f"b{'0'*(binValueFullWidth - len(binaryValue))}{binaryValue}"
            if finalValueFormat.lower() == 'binary':
                finalValue = binaryValue
            elif finalValueFormat.lower() == 'hex':
                if binaryValue[0] == 'b':
                    finalValue = hex(int(binaryValue[1:], 2))[2:]  # TODO: Identify cases where binary numbers have internal floating signals
                    if extendBinaryValuesToWidth:
                        binValueFullWidth = vcdReader1.identifierWidthDictionary[identifier]
                        finalValue = "0x"+f"{'0'*((binValueFullWidth // 4) - len(finalValue))}{finalValue}".upper()
                else:
                    finalValue = binaryValue
            else:
                finalValue = binaryValue
            # END Value modification subsection

            signalTypes = vcdReader1.identifierTypeDictionary[identifier]
            signalTypeStr = list()
            for signalType in signalTypes:
                signalTypeStr.append(signalType.value)
            signalNames = list(dict.fromkeys(vcdReader1.identifierDictionary[identifier]))
            signalNames = signalNames if len(signalNames) > 1 else signalNames[0]
            signalTypeStr = signalTypeStr if len(signalTypeStr) > 1 else signalTypeStr[0]
            translatedLine = f"Line {i+1}: {signalTypeStr} {signalNames} {identifier} {finalValue}"
            finishedLines.append(translatedLine)

        else:  # potentialMatch is None:
            potentialDelayMatch = re.match(vcdReader1.compiledVCDDelayPattern, line)
            if potentialDelayMatch is not None:
                if includeDelayValuesAndTimestamps:
                    delayValue = potentialDelayMatch.groups()[0]
                    if includeShapValueList and len(shapValueList) != 0:  # Assume correct shap list is provided
                        translatedLine = f"Line {i+1}: {delayValue} - HW_TS{counter} - SHAP: [{shapValueList[counter]}] "
                    else:
                        translatedLine = f"Line {i+1}: {delayValue} - HW_TS{counter} "
                    if len(translatedLine) < timestampLineVisualFillToLength:
                        translatedLine += (timestampLineVisualFillToLength - len(translatedLine)) * "-"
                    finishedLines.append(translatedLine)
                if stopAtTimestamp is not None:
                    if counter == stopAtTimestamp:
                        break
                counter += 1

    with open(savedTranslatedVCDLinesPath, "w") as outFile:
        for line in finishedLines:
            outFile.write(f"{line}\n")
