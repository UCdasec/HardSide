#!/bin/bash

for i in {1..50000}; do
  old_filename="/home/muhibkhan/tofu-master/50000_traces/VHDL/50000_test_vhdl_HW/aes${i}.vcd"
  new_filename="/home/muhibkhan/tofu-master/50000_traces/VHDL/50000_test_vhdl_HW/plaintext${i}.vcd"
  mv "$old_filename" "$new_filename"
done
