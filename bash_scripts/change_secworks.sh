#!/bin/bash

file_path="/home/muhibkhan/SecWorks_AES_Impl/SecWorks_AES_Impl.srcs/sources_1/imports/src/tb/tb_aes.v"
plaintext_file="/home/muhibkhan/Plaintexts/plaintexts3.txt"
line_number=436
iterations=20

for i in $(seq 1 $iterations); do
    new_content=$(sed -n "${i}p" "$plaintext_file")
    new_line="      nist_plaintext0 = 128'h$new_content;"
    sed -i "${line_number}s/.*/$new_line/" "$file_path"
    echo "Replaced line $line_number with: $new_line"
    
    # Change line 445 in tb_aes.v file
    new_dumpfile="\"plaintext$((1000 + i)).vcd\""
    sed -i "445s/.*/\    \$dumpfile($new_dumpfile);/" "$file_path"
    echo "Changed line 445 to: \$dumpfile($new_dumpfile);"
done

