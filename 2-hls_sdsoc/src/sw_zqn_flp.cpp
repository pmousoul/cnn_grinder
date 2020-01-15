// ZynqNet SW floating-point implementation

#include "sqj2_tb.hpp"

void sw_zqn_flp(
	float* sw_fmap_flp[4], float* sw_params_flp[ZQN_LN], float* sw_fmap_flp_o, Largs_zqn& larg, float error_sz, bool* error_flag )
{
	// Count total convolution time
	stopwatch timer;
	double latency[1+ZQN_LN+15];

	// Layer index
	uint8_t y=0;
	uint8_t i=0;

	std::string implementation = "sw_zqn_flp:";

	// Reshape the input from 256x256x3 to 128x128x32
	timer.start();
	reshape_input<float>( sw_fmap_flp[0], sw_fmap_flp[1], 256, 256, 3, 32, 3, 2, 1 );
	timer.stop();
	latency[y++] = timer.duration();

	// C1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "CONV1",
			"2_conv1.bin", "data/float/zqn/2_conv1.bin",
			error_sz, error_flag );
	#endif

	// fire2
	// F2S3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F2E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F2E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE2",
			"3_fire2.bin", "data/float/zqn/3_fire2.bin",
			error_sz, error_flag );
	#endif

	// fire3
	// F3S1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F3E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F3E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE3",
			"4_fire3.bin", "data/float/zqn/4_fire3.bin",
			error_sz, error_flag );
	#endif

	// fire4
	// F4S3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F4E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F4E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE4",
			"5_fire4.bin", "data/float/zqn/5_fire4.bin",
			error_sz, error_flag );
	#endif

	// fire5
	// F5S1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F5E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F5E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE5",
			"6_fire5.bin", "data/float/zqn/6_fire5.bin",
			error_sz, error_flag );
	#endif

	// fire6
	// F6S3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F6E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F6E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE6",
			"7_fire6.bin", "data/float/zqn/7_fire6.bin",
			error_sz, error_flag );
	#endif

	// fire7
	// F7S1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F7E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F7E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE7",
			"8_fire7.bin", "data/float/zqn/8_fire7.bin",
			error_sz, error_flag );
	#endif

	// fire8
	// F8S3_1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F8S3_2
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[1], larg.h_in[i+1], larg.w_in[i+1], larg.ch_out[i-1], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F8E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F8E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[3], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE8",
			"9_fire8.bin", "data/float/zqn/9_fire8.bin",
			error_sz, error_flag );
	#endif

	// fire9
	// F9S1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F9E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F9E3_1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// F9E3_2
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[3], sw_fmap_flp[0], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i-1] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[1], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i-2] , larg.ch_out[i-2], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE9",
			"10_fire9.bin", "data/float/zqn/10_fire9.bin",
			error_sz, error_flag );
	#endif

	// C10_11
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_12
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[1], sw_fmap_flp[2], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_21
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_22
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[1], sw_fmap_flp[2], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[3], sw_fmap_flp[0], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], 2*larg.ch_out[i] , 2*larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[1], ZQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "CONV10",
			"11_conv10.bin", "data/float/zqn/11_conv10.bin",
			error_sz, error_flag );
	#endif

	// Avgpool
	timer.start();
	sw_avgpool_flp(
		sw_fmap_flp[1], sw_fmap_flp_o, larg.h_in[i], larg.w_in[i], 1024, 0, 8, 1 );
	timer.stop();
	latency[y++] = timer.duration();

	// Softmax
	timer.start();
	softmax(sw_fmap_flp_o, 1024);
	timer.stop();
	latency[y++] = timer.duration();

	// Print propabilities
	DEBUG( "ZynqNet floating-point SW implementation - Top-5 results:" );
	DEBUG( "----------------------------------------------------------" );
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

}
