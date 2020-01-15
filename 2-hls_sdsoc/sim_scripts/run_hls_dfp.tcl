open_project sqj2_test_dfp

set_top hw_conv_mpool_dfp

add_files     ../src/hw_conv_mpool_flp.cpp -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"

add_files     ../src/hw_conv_mpool_dfp.cpp -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"

add_files -tb ../src/sqj2_tb.cpp           -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/sqj2_tb_helper.cpp    -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/test_sqn_dfp.cpp      -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/test_zqn_dfp.cpp      -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/test_zqn_flp.cpp      -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"

add_files -tb ../src/sw_sqn_flp.cpp        -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/sw_sqn_dfp.cpp        -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/hw_sqn_dfp.cpp        -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"

add_files -tb ../src/sw_zqn_flp.cpp        -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/sw_zqn_dfp.cpp        -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/hw_zqn_dfp.cpp        -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"
add_files -tb ../src/hw_zqn_flp.cpp        -cflags "-std=c++0x -O3 -D__SIM__ -DUSEDEBUG -I../src/include"

add_files -tb ../src/data/dfixed/sqn/image_dfixed.bin
add_files -tb ../src/data/dfixed/sqn/params_dfixed.bin

add_files -tb ../src/data/dfixed/sqn/3_pool1.bin
add_files -tb ../src/data/dfixed/sqn/4_fire2.bin
add_files -tb ../src/data/dfixed/sqn/6_pool3.bin
add_files -tb ../src/data/dfixed/sqn/7_fire4.bin
add_files -tb ../src/data/dfixed/sqn/9_pool5.bin
add_files -tb ../src/data/dfixed/sqn/10_fire6.bin
add_files -tb ../src/data/dfixed/sqn/11_fire7.bin
add_files -tb ../src/data/dfixed/sqn/12_fire8.bin
add_files -tb ../src/data/dfixed/sqn/13_fire9.bin
add_files -tb ../src/data/dfixed/sqn/14_conv10.bin

add_files -tb ../src/data/class_labels.txt


open_solution "solution1"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 10 -name default

csim_design -argv {0.0000001} -clean -O -compiler gcc
csynth_design
cosim_design -argv {0.0000001}

exit
