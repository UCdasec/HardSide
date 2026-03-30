#!/bin/bash

# Initial setup
for ((i=12001; i<=50000; i++))
do
    current_filename="plaintext$i"
    next_filename="plaintext$((i+1))"

    # Run emptyline.py
    #python3 /home/muhibkhan/emptyline.py /home/muhibkhan/tofu-master/501to1000_test_HW/$current_filename.vcd

# Modify JSON file
    json_file="/home/muhibkhan/tofu-master/50000_traces/SecWorks/50000_test_secworks_HW/settings_example.json"
    sed -i "s/\"vcdGlob\": \"$current_filename.vcd\",/\"vcdGlob\": \"$next_filename.vcd\",/" "$json_file"
    sed -i "s/\"pickleGlob\": \"$current_filename.pickle\",/\"pickleGlob\": \"$next_filename.pickle\",/" "$json_file"
    sed -i "s/\"traceFileName\": \"$current_filename.h5\",/\"traceFileName\": \"$next_filename.h5\",/" "$json_file"
    
    cd /home/muhibkhan/tofu-master

    python3 parse.py --settings 50000_traces/SecWorks/50000_test_secworks_HW/settings_example.json

    python3 synthesize.py --settings 50000_traces/SecWorks/50000_test_secworks_HW/settings_example.json

    #cd ..

done

#python3 /home/muhibkhan/emptyline.py /home/muhibkhan/tofu-master/500_test/plaintext1000.vcd
#cd tofu-master
#python3 parse.py --settings 500_test/settings_example.json
#python3 synthesize.py --settings 500_test/settings_example.json
#cd
