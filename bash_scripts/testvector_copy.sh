#!/bin/bash

main_file="/home/muhibkhan/Plaintexts/50000_plaintexts.txt"

for i in {1..50000}; do
    input_file="/home/muhibkhan/tofu-master/example-aes-vhdl/testvector/testvector${i}.tv"
    sed "${i}q;d" "$main_file" | awk '{print $0 " d2d5016882839143969ee9a253a752e1"}' >> "$input_file"
done

