#!/bin/bash

# Vivado HLS
#source /mnt/terabyte/pmousoul_data/sw/Xilinx/Vivado/2017.2/settings64.sh

# Vivado HLS_SDx
/home/pmousoul/Desktop/pmousoul_data/sw/Xilinx_SDx/SDx/2018.2/settings64.sh

# Select accelerator to simulate
vivado_hls -f run_hls_dfp.tcl
#vivado_hls -f run_hls_flp.tcl

