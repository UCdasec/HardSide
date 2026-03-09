# # Modify tb_aes.v file with plaintexts and run simulations
# file_path="/home/muhibkhan/picorv32/firmware/aes.c"
# plaintext_file="/home/muhibkhan/Plaintexts/satisfya.txt"
# tb_file="/home/muhibkhan/picorv32/testbench.v"
# line_number=558
# iterations=2

# for i in $(seq 1 $iterations); do
#     new_content=$(sed -n "${i}p" "$plaintext_file")
#     new_line="    uint8_t input[16] = {$new_content};"
#     sed -i "${line_number}s/.*/$new_line/" "$file_path"
#     echo "Replaced line $line_number with: $new_line"


#     new_dumpfile="\"plaintext$((i)).vcd\""
#     sed -i "28s/.*/\    		\$dumpfile($new_dumpfile);/" "$tb_file"
#     echo "Changed line 28 to: \$dumpfile($new_dumpfile);"
    
#     cd /home/muhibkhan/picorv32
#     sudo make clean
#     sudo make testbench_verilator
#     sudo make test_vcd
#  done


file_path="/home/muhibkhan/picorv32/firmware/aes.c"
plaintext_file="/home/muhibkhan/Plaintexts/plaintexts_0xaa_format.txt"
tb_file="/home/muhibkhan/picorv32/testbench.v"
line_number=558
start_line=30001
iterations=2

for i in $(seq 0 $((iterations-1))); do
    current_line=$((start_line + i))
    new_content=$(sed -n "${current_line}p" "$plaintext_file")
    new_line="    uint8_t input[16] = {$new_content};"
    sed -i "${line_number}s/.*/$new_line/" "$file_path"
    echo "Replaced line $line_number with: $new_line"

    new_dumpfile="\"plaintext$((start_line + i)).vcd\""
    sed -i "28s/.*/\    		\$dumpfile($new_dumpfile);/" "$tb_file"
    echo "Changed line 28 to: \$dumpfile($new_dumpfile);"
    
    cd /home/muhibkhan/picorv32
    sudo make clean
    sudo make testbench_verilator
    sudo make test_vcd
done