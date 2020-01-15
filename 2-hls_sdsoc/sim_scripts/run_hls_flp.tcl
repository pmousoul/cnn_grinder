open_project sqj2_test_flp

set_top hw_conv_mpool_flp

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

add_files -tb ../src/data/float/zqn/image_float.bin
add_files -tb ../src/data/float/zqn/params_float.bin

add_files -tb ../src/data/float/zqn/2_conv1.bin
add_files -tb ../src/data/float/zqn/3_fire2.bin
add_files -tb ../src/data/float/zqn/4_fire3.bin
add_files -tb ../src/data/float/zqn/5_fire4.bin
add_files -tb ../src/data/float/zqn/6_fire5.bin
add_files -tb ../src/data/float/zqn/7_fire6.bin
add_files -tb ../src/data/float/zqn/8_fire7.bin
add_files -tb ../src/data/float/zqn/9_fire8.bin
add_files -tb ../src/data/float/zqn/10_fire9.bin
add_files -tb ../src/data/float/zqn/11_conv10.bin

add_files -tb ../src/data/class_labels.txt


open_solution "solution1"
set_part {xc7z045ffg900-2} -tool vivado
create_clock -period 10 -name default

csim_design -argv {0.1} -clean -O -compiler gcc
csynth_design
cosim_design -argv {0.1}

exit
