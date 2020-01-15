// SqN v1.1 SW floating-point implementation

#include "sqj2_tb.hpp"

void sw_sqn_flp(
	float* sw_fmap_flp[4], float* sw_params_flp[SQN_LN], Largs_sqn& larg, float error_sz, bool* error_flag )
{
	// Count total convolution time
	stopwatch timer;
	double latency[1+SQN_LN+18];

	// Layer index
	uint8_t y=0;
	uint8_t i=0;

	std::string implementation = "sw_sqn_flp:";

	// Reshape the input from 227x227x3 to 113x113x32
	timer.start();
	reshape_input<float>( sw_fmap_flp[0], sw_fmap_flp[1], 227, 227, 3, 32, 3, 2, 0 );
	timer.stop();
	latency[y++] = timer.duration();

	// conv1 - pool1
	// C1_MP1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();

	timer.start();
	sw_maxpool<float>(
		sw_fmap_flp[0], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i],
		larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[1], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "C1_MP1",
			"3_pool1.bin", "data/float/sqn/3_pool1.bin",
			error_sz, error_flag );
	#endif

	// fire2
	// F2S
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F2E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F2E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[1], sw_fmap_flp[2], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE2",
			"4_fire2.bin", "data/float/sqn/4_fire2.bin",
			error_sz, error_flag );
	#endif

	// fire3 - pool3
	// F3S
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F3E1_MP3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	sw_maxpool<float>(
		sw_fmap_flp[0], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i], larg.ch_out[i],
		larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F3E3_MP3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	sw_maxpool<float>(
		sw_fmap_flp[0], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i],
		larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[1], sw_fmap_flp[0], larg.h_in[i+1], larg.w_in[i+1], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "F3_MP3",
			"6_pool3.bin", "data/float/sqn/6_pool3.bin",
			error_sz, error_flag );
	#endif

	// fire4
	// F4S
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
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F4E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[0], sw_fmap_flp[2], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[1], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE4",
			"7_fire4.bin", "data/float/sqn/7_fire4.bin",
			error_sz, error_flag );
	#endif

	// fire5 - pool5
	// F5S
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F5E1_MP5
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	sw_maxpool<float>(
		sw_fmap_flp[1], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i], larg.ch_out[i],
		larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F5E3_MP5
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	sw_maxpool<float>(
		sw_fmap_flp[1], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i],
		larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[2], sw_fmap_flp[0], sw_fmap_flp[1], larg.h_in[i+1], larg.w_in[i+1], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[1], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "F5_MP5",
			"9_pool5.bin", "data/float/sqn/9_pool5.bin",
			error_sz, error_flag );
	#endif

	// fire6
	// F6S
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F6E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F6E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[1], sw_fmap_flp[2], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE6",
			"10_fire6.bin", "data/float/sqn/10_fire6.bin",
			error_sz, error_flag );
	#endif

	// fire7
	// F7S
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
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F7E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[0], sw_fmap_flp[2], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[1], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE7",
			"11_fire7.bin", "data/float/sqn/11_fire7.bin",
			error_sz, error_flag );
	#endif

	// fire8
	// F8S
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F8E1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F8E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[0], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[1], sw_fmap_flp[2], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE8",
			"12_fire8.bin", "data/float/sqn/12_fire8.bin",
			error_sz, error_flag );
	#endif

	// fire9
	// F9S
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
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// F9E3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[0], sw_fmap_flp[2], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[1], SQN_FMAP_MAX_SZ, 0,
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE9",
			"13_fire9.bin", "data/float/sqn/13_fire9.bin",
			error_sz, error_flag );
	#endif

	// C10_1
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// C10_2
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[0], sw_fmap_flp[2], sw_fmap_flp[3], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// C10_3
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;

	// C10_4
	timer.start();
	sw_conv_flp(
		sw_fmap_flp[1], sw_params_flp[i], sw_fmap_flp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i], 1 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[0], sw_fmap_flp[2], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels(
		sw_fmap_flp[3], sw_fmap_flp[1], sw_fmap_flp[0], larg.h_in[i], larg.w_in[i], 2*larg.ch_out[i] , 1000-2*larg.ch_out[i], 2*larg.ch_out[i] - (1000-2*larg.ch_out[i]) );
	timer.stop();
	latency[y++] = timer.duration();

	#ifdef USEDEBUG
		check_iresults<float>(
			sw_fmap_flp[0], SQN_FMAP_MAX_SZ, 0,
			1, 1, 1000, implementation + "CONV10",
			"14_conv10.bin", "data/float/sqn/14_conv10.bin",
			error_sz, error_flag );
	#endif

	// Avgpool
	timer.start();
	sw_avgpool_flp(
		sw_fmap_flp[0], sw_fmap_flp[1], larg.h_in[i], larg.w_in[i], 1000, 0, 14, 1 );
	timer.stop();
	latency[y++] = timer.duration();

	// Softmax
	timer.start();
	softmax(sw_fmap_flp[1], 1000);
	timer.stop();
	latency[y++] = timer.duration();

	// Print propabilities
	DEBUG( "SqN v1.1 floating-point SW implementation - Top-5 results:" );
	DEBUG( "----------------------------------------------------------" );
#ifdef USEDEBUG
#if defined(__SIM__)
	print_probs("class_labels.txt", sw_fmap_flp[1], 1000);
#else
	print_probs("data/class_labels.txt", sw_fmap_flp[1], 1000);
#endif
#endif

	// Print latencies
	double total = 0;
	for( int n=0; n<(1+SQN_LN+18); n++)
	{
		DEBUG( "layer " << std::right << std::setw(2) << larg.layer_name[n] << "\t latency (sec): " << latency[n] );
		total += latency[n];
	}
	DEBUG( "TOTAL latency (sec): " << total << std::endl );

}
