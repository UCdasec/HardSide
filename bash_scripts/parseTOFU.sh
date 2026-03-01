#!/bin/bash
#parseTOFU.sh
# By Muhib Khan
# DaSec Lab, University of Cincinnati

# Initial setup
for ((i=1; i<=5000; i++))
do
    current_filename="plaintext$i"
    next_filename="plaintext$((i+1))"

# Modify JSON file
    json_file="/home/username/tofu-master/50000_traces/SecWorks/50000_test_secworks_HW/settings_example.json"
    sed -i "s/\"vcdGlob\": \"$current_filename.vcd\",/\"vcdGlob\": \"$next_filename.vcd\",/" "$json_file"
    sed -i "s/\"pickleGlob\": \"$current_filename.pickle\",/\"pickleGlob\": \"$next_filename.pickle\",/" "$json_file"
    sed -i "s/\"traceFileName\": \"$current_filename.h5\",/\"traceFileName\": \"$next_filename.h5\",/" "$json_file"
    
    cd /home/username/tofu-master

    python3 parse.py --settings 50000_traces/SecWorks/50000_test_secworks_HD/settings_example.json


    python3 synthesize.py --settings 50000_traces/SecWorks/50000_test_secworks_HD/settings_example.json
    
done
