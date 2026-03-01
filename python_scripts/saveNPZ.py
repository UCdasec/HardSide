import h5py
import os
import numpy as np
output_path='/home/username/ProjectVault_HD_NPZ/' 
os.makedirs(output_path, exist_ok=True)

file_name='ProjectVault_5000_HD.npz'                                          

outpath = os.path.join(output_path,file_name)

text_file_path = '/home/username/Plaintexts/plaintext5000_decimal.txt'
with open(text_file_path, 'r') as file:
    plaintexts = file.readlines()
plain_text = np.array([list(map(int, line.split())) for line in plaintexts], dtype=np.uint16)

# key used in the simulation for generating all the VCD files. 
static_key = [210, 213, 1, 104, 130, 131, 145, 67, 150, 158, 233, 162, 83, 167, 82, 225]
key = np.array(static_key, dtype=np.uint8)

# the path where all the h5 files are stored. 
file_pattern = 'home/username/tofu-master/ProjectVault_5k_HD/plaintext{}.h5'
all_trace_arrays = []
for i in range(1, 4):
    file_path = file_pattern.format(i)    
    with h5py.File(file_path, 'r') as h5_file:
        trace_array = h5_file['leakages'][:]
        all_trace_arrays.append(trace_array)
power_trace = np.concatenate(all_trace_arrays, axis=0)

print(f"Shape of Power Trace: {np.shape(power_trace)}")
print(power_trace)
print(f"Shape of Plaintexts: {np.shape(plain_text)}")
print(plain_text)
print(f"Shape of Key: {np.shape(key)}")
print(key)

np.savez(outpath, power_trace=power_trace, plain_text=plain_text, key=key)
