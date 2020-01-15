# ZynqNet SDSoC 2018.2 Implementation

## About
We modified the ZynqNet HLS code found in:
https://github.com/dgschwend/zynqnet/tree/master/_HLS_CODE
and made it compatible with Xilinx SDSoC 2018.2.


## Modifications

* SDSoC compatible interface definition in the `src/fpga_top.hpp` file:
    #pragma SDS data mem_attribute(SHARED_DRAM:PHYSICAL_CONTIGUOUS)
    #pragma SDS data access_pattern(SHARED_DRAM:SEQUENTIAL)
    #pragma SDS data zero_copy(SHARED_DRAM[0:DRAM_DEPTH])

* Disabled a function inlining in order to make the design synthesizable in SDSoC 2018.2 (https://github.com/dgschwend/zynqnet/issues/54):
    void preloadPixelFromDRAM(data_t *SHARED_DRAM) {
    #pragma HLS inline off


## Build the project in SDSoC 2018.2

After the `zynqnet` (the name is user defined) Linux application project has been created, the user must copy the `src` folder contents to the `src` folder of the SDSoC folder.

In the SDSoC tool the user must:

(1) Set the Active Build Configuration to `Release`.

(2) Change the following `zynqnet` project properties:

* Go to `C/C++ Build -> SDS++ Compiler -> Optimization` and set the following:
	* Optimization Level: `Optimize most (-O3)`
	* Other optimization flags: `-std=c++0x -Wno-unused-label`

(3) Set the desired "HW" function for acceleration, e.g. expand the `fpga_top.cpp` file and select the `fpga_top()` function for acceleration - right-click on the function and select `Toggle HW/SW`.

(4) Set the desired frequency for the accelerator and the data-mover.

(5) Right-click on the project and select `Build`.
