// ZynqNet SW fixed-point implementation

#include "sqj2_tb.hpp"


void sw_zqn_dfp(
	int8_t* sw_fmap_dfp[4], int8_t* sw_params_dfp[ZQN_LN], float* sw_fmap_flp_o, Largs_zqn& larg, float error_sz, bool* error_flag )
{
	// Count total convolution time
	stopwatch timer;
	double latency[1+ZQN_LN+15];

	// Allocate memory for float layer io
	float* sw_fmap_flp = new float[ZQN_FMAP_MAX_SZ];

	// Load layer arguments
	int8_t ei[29] = { 0,   -3, -5, -5,   -6, -6, -6,   -7, -8, -8,   -7, -8, -8,   -7, -7, -7,   -7, -7, -7,   -6, -6, -6, -6,   -5, -4, -4, -4,   -2, -2 };
	int8_t eo[29] = {-3,   -5, -6, -6,   -6, -7, -7,   -8, -7, -7,   -8, -7, -7,   -7, -7, -7,   -7, -6, -6,   -6, -6, -5, -5,   -4, -2, -2, -2,   -1, -1 };
	int8_t ep[29] = { 7,    6,  6,  7,    6,  7,  7,    8,  7,  8,    7,  7,  7,    8,  7,  8,    7,  8,  8,    8,  8,  8,  8,    8,  8,  9,  9,   10, 10 };

	// Layer indices
	uint8_t y=0;
	uint8_t i=0;
	uint8_t ii=0;

	std::string implementation = "sw_zqn_dfp:";

	// Reshape the input from 256x256x3 to 128x128x32
	timer.start();
	reshape_input<int8_t>( sw_fmap_dfp[0], sw_fmap_dfp[1], 256, 256, 3, 32, 3, 2, 1 );
	timer.stop();
	latency[y++] = timer.duration();

	// C1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "CONV1",
			"2_conv1.bin", "data/dfixed/zqn/2_conv1.bin",
			error_sz, error_flag );
	#endif

	// fire2
	// F2S3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F2E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F2E3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE2",
			"3_fire2.bin", "data/dfixed/zqn/3_fire2.bin",
			error_sz, error_flag );
	#endif

	// fire3
	// F3S1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F3E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F3E3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE3",
			"4_fire3.bin", "data/dfixed/zqn/4_fire3.bin",
			error_sz, error_flag );
	#endif

	// fire4
	// F4S3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F4E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F4E3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE4",
			"5_fire4.bin", "data/dfixed/zqn/5_fire4.bin",
			error_sz, error_flag );
	#endif

	// fire5
	// F5S1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F5E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F5E3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE5",
			"6_fire5.bin", "data/dfixed/zqn/6_fire5.bin",
			error_sz, error_flag );
	#endif

	// fire6
	// F6S3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F6E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F6E3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE6",
			"7_fire6.bin", "data/dfixed/zqn/7_fire6.bin",
			error_sz, error_flag );
	#endif

	// fire7
	// F7S1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F7E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F7E3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE7",
			"8_fire7.bin", "data/dfixed/zqn/8_fire7.bin",
			error_sz, error_flag );
	#endif

	// fire8
	// F8S3_1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F8S3_2
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[1], larg.h_in[i+1], larg.w_in[i+1], larg.ch_out[i-1], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F8E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F8E3
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[3], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE8",
			"9_fire8.bin", "data/dfixed/zqn/9_fire8.bin",
			error_sz, error_flag );
	#endif

	// fire9
	// F9S1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F9E1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F9E3_1
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F9E3_2
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[1], sw_params_dfp[i], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[3], sw_fmap_dfp[0], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i-1] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[2], sw_fmap_dfp[1], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i-2] , larg.ch_out[i-2], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef USEDEBUG
		check_iresults<int8_t>(
			sw_fmap_dfp[0], ZQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE9",
			"10_fire9.bin", "data/dfixed/zqn/10_fire9.bin",
			error_sz, error_flag );
	#endif

	// C10_11
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_12
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[1], sw_fmap_dfp[2], sw_fmap_dfp[3], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// C10_21
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_22
	timer.start();
	sw_conv_dfp(
		sw_fmap_dfp[0], sw_params_dfp[i], sw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[1], sw_fmap_dfp[2], sw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		sw_fmap_dfp[3], sw_fmap_dfp[0], sw_fmap_dfp[1], larg.h_in[i], larg.w_in[i], 2*larg.ch_out[i] , 2*larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();

	// Convert C10 result to float
	for(int k=0; k < (larg.h_in[i]*larg.w_in[i]*1024); k++ )
		sw_fmap_flp[k] = (float) ( sw_fmap_dfp[1][k] ) * exp2(-eo[ii]);

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp, ZQN_FMAP_MAX_SZ, 0,
			8, 8, 1024, implementation + "CONV10",
			"11_conv10.bin", "data/dfixed/zqn/11_conv10.bin",
			error_sz, error_flag );
	#endif

	// Avgpool
	timer.start();
	sw_avgpool_flp(
		sw_fmap_flp, sw_fmap_flp_o, larg.h_in[i], larg.w_in[i], 1024, 0, 8, 1 );
	timer.stop();
	latency[y++] = timer.duration();

	// Softmax
	timer.start();
	softmax(sw_fmap_flp_o, 1024);
	timer.stop();
	latency[y++] = timer.duration();

	// Print propabilities
	DEBUG( "ZynqNet fixed-point SW implementation - Top-5 results:" );
	DEBUG( "-------------------------------------------------------" );
#ifdef USEDEBUG
#if defined(__SIM__)
	print_probs("class_labels.txt", sw_fmap_flp_o, 1024);
#else
	print_probs("data/class_labels.txt", sw_fmap_flp_o, 1024);
#endif
#endif

	// Print latencies
	double total = 0;
	for( int n=0; n<(1+ZQN_LN+15); n++)
	{
		DEBUG( "layer " << std::right << std::setw(2) << larg.layer_name[n] << "\t latency (sec): " << latency[n] );
		total += latency[n];
	}
	DEBUG( "TOTAL latency (sec): " << total << std::endl );


	// Free memory
	delete[] sw_fmap_flp;

}
