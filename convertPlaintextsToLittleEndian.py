# convertPlaintextsToLittleEndian.py - Logan Reichling - Start 9/12/25 - UC DaSec
# Converts a list of hex plaintexts to little endian format
import os

if __name__ == "__main__":
    plaintextFilepath = r"C:\Users\Logan Reichling\Desktop\HardSide Project\Old method VCD Generation\50000_plaintexts.txt"
    outputFilepath = r"C:\Users\Logan Reichling\Desktop\HardSide Project\Old method VCD Generation\50000_plaintexts_little_endian.txt"

    # Ensure input file path exists:
    if not os.path.exists(plaintextFilepath):
        if not os.path.isfile(plaintextFilepath):
            print(f"Input file path does not exist: {plaintextFilepath}")
            exit(1)
    dirName, fileName = os.path.split(plaintextFilepath)

    with open(plaintextFilepath, 'r') as inFile:
        origPlaintextLines = inFile.readlines()

    lePlaintextLines = list()
    for i, pt in enumerate(origPlaintextLines):
        pt = pt.strip()
        temp = ""
        if len(pt) != 32:
            print(f"Plaintext incorrect length at line {i}, {len(pt)} != 32")
            exit(1)
        for j in range(32, 0, -2):
            temp += pt[j - 2:j]
        temp += '\n'
        lePlaintextLines.append(temp)

    with open(outputFilepath, 'w') as outfile:
        outfile.writelines(lePlaintextLines)