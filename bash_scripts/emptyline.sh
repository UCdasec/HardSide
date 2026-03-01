#!/bin/bash

# Initial setup
for ((i=1; i<50001; i++))
do
    current_filename="plaintext$i"
    #next_filename="plaintext$((i+1))"

    # Run emptyline.py
    python3 /home/username/emptyline.py /home/username/tofu-master/50000_traces/SecWorks/50000_test_secworks_HW/$current_filename.vcd

done
