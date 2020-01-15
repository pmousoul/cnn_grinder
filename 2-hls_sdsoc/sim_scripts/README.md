## SqueezeJet-2 HLS simulation scripts

* In the `run_all.sh` set the Vivado HLS location of the `settings64.sh` and source it. Also select the floating-point or the dynamic fixed-point accelerator version to simulate. Due to Vivado HLS (version 2018.2) simulator bug(s) the floating-point version requires 3 days and 40-50GB of ram to complete on a contemporary (40-core) server (with Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz CPUs).

* Edit the `../src/include/hw_func_*.hpp` files and set the appropriate location for the following file: `Xilinx_SDx/Vivado/2018.2/include/gmp.h` (Maybe this is not required but it is carried from older Vivado HLS versions.)

* Make sure that the correct function test is uncommented in the `2-hls_sdsoc/src/sqj2_tb.cpp` file. For example, uncomment the `return test_sqn_dfp(error_sz, cpuf);` statement to test the dynamic fixed-point version of SqueezeJet-2.