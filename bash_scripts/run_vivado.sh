#!/bin/bash

VIVADO_PATH="/home/muhibkhan/Vivado/2023.2"
source "$VIVADO_PATH/settings64.sh"

# Navigate to the project directory
cd /home/muhibkhan/Desktop/SecWorks_AES_Impl

# Modify tb_aes.v file with plaintexts and run simulations
file_path="/home/muhibkhan/Desktop/SecWorks_AES_Impl/SecWorks_AES_Impl.srcs/sources_1/imports/src/tb/tb_aes.v"
plaintext_file="/home/muhibkhan/Plaintexts/plaintexts12001to50000.txt"
line_number=449
iterations=38000

for i in $(seq 1 $iterations); do
    new_content=$(sed -n "${i}p" "$plaintext_file")
    new_line="      nist_plaintext0 = 128'h$new_content;"
    sed -i "${line_number}s/.*/$new_line/" "$file_path"
    echo "Replaced line $line_number with: $new_line"
    
    # Change line 446 in tb_aes.v file
    new_dumpfile="\"plaintext$((12000 + i)).vcd\""
    sed -i "542s/.*/\    \$dumpfile($new_dumpfile);/" "$file_path"
    echo "Changed line 542 to: \$dumpfile($new_dumpfile);"

    # Run simulations with a delay of 10 microseconds
    vivado -mode tcl -source launch_simulation.tcl
done
