# gen_plaintexts.py - Logan Reichling - UC DaSec
# Generate random plaintexts for testing
from tqdm import tqdm
import secrets

with open("200k_plaintexts.txt", "w") as outFile:
    for i in tqdm(range(200000)):
        outFile.write(f"{secrets.token_bytes(16).hex()}\n")
