// SqN v1.1 HW dynamic fixed-point implementation

#include "sqj2_tb.hpp"
#include "hw_func_dfp.hpp"


void hw_sqn_dfp(
	p_fmap_t* hw_fmap_dfp[4], p_param_t* hw_params_dfp[SQN_LN], float* hw_fmap_flp_o, Largs_sqn& larg, float error_sz, bool* error_flag )
{
	// Count total convolution time
	stopwatch timer;
	double latency[1+SQN_LN+13] = { 0 };

	// Allocate memory for float layer io
	float* hw_fmap_flp = new float[SQN_FMAP_MAX_SZ];

	// Load layer arguments
	int8_t ei[26] = { 0,   -3, -4, -4,   -3, -3, -3,   -3, -4, -4,   -4, -4, -4,   -3, -4, -4,   -3, -4, -4,   -3, -3, -3,   -3, -3, -3,   -2};
	int8_t eo[26] = {-3,   -4, -3, -3,   -3, -3, -3,   -4, -4, -4,   -4, -3, -3,   -4, -3, -3,   -4, -3, -3,   -3, -3, -3,   -3, -2, -2,   -1};
	int8_t ep[26] = { 7,    6,  7,  7,    7,  7,  7,    6,  7,  7,    7,  7,  7,    7,  8,  7,    7,  8,  8,    7,  7,  7,    8,  7,  8,    8};

	// Layer indices
	uint8_t y=0;
	uint8_t i=0;
	uint8_t ii=0;

	std::string implementation = "hw_sqn_dfp:";

	// Reshape the input from 227x227x3 to 113x113x32
	timer.start();
	reshape_input<int8_t>( (int8_t*)hw_fmap_dfp[0], (int8_t*)hw_fmap_dfp[1], 227, 227, 3, 32, 3, 2, 0 );
	timer.stop();
	latency[y++] = timer.duration();

	// conv1 - pool1
	// C1_MP1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "C1_MP1",
			"3_pool1.bin", "data/dfixed/sqn/3_pool1.bin",
			error_sz, error_flag );
	#endif

	// fire2
	// F2S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F2E1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F2E3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i] , larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE2",
			"4_fire2.bin", "data/dfixed/sqn/4_fire2.bin",
			error_sz, error_flag );
	#endif

	// fire3 - pool3
	// F3S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F3E1_MP3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F3E3_MP3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i+1], larg.w_in[i+1], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "F3_MP3",
			"6_pool3.bin", "data/dfixed/sqn/6_pool3.bin",
			error_sz, error_flag );
	#endif

	// fire4
	// F4S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F4E1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F4E3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE4",
			"7_fire4.bin", "data/dfixed/sqn/7_fire4.bin",
			error_sz, error_flag );
	#endif

	// fire5 - pool5
	// F5S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F5E1_MP5
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F5E3_MP5
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i+1], larg.w_in[i+1], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "F5_MP5",
			"9_pool5.bin", "data/dfixed/sqn/9_pool5.bin",
			error_sz, error_flag );
	#endif

	// fire6
	// F6S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F6E1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F6E3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE6",
			"10_fire6.bin", "data/dfixed/sqn/10_fire6.bin",
			error_sz, error_flag );
	#endif

	// fire7
	// F7S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F7E1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F7E3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE7",
			"11_fire7.bin", "data/dfixed/sqn/11_fire7.bin",
			error_sz, error_flag );
	#endif

	// fire8
	// F8S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F8E1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F8E3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE8",
			"12_fire8.bin", "data/dfixed/sqn/12_fire8.bin",
			error_sz, error_flag );
	#endif

	// fire9
	// F9S
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F9E1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;
	// F9E3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[1], hw_params_dfp[i], hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++; ii++;

	#ifdef TRACE
	sleep(TRACE_TIME);
	#endif
	#ifdef USEDEBUG
		check_iresults<int8_t>(
			(int8_t*)hw_fmap_dfp[0], SQN_FMAP_MAX_SZ, -ei[ii],
			larg.h_in[i], larg.w_in[i], larg.ch_in[i], implementation + "FIRE9",
			"13_fire9.bin", "data/dfixed/sqn/13_fire9.bin",
			error_sz, error_flag );
	#endif

	// C10_1
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_2
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[1], (int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[3], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_3
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	i++;
	// C10_4
	timer.start();
	hw_conv_mpool_dfp(
		hw_fmap_dfp[0], hw_params_dfp[i], hw_fmap_dfp[2], larg.h_in[i], larg.w_in[i],
		larg.ch_in[i], larg.ch_out[i], larg.pad[i], larg.kernel[i], larg.stride[i],
		ei[ii], eo[ii], ep[ii], 1, larg.ppad[i], larg.pkernel[i], larg.pstride[i] );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[1], (int8_t*)hw_fmap_dfp[2], (int8_t*)hw_fmap_dfp[0], larg.h_in[i], larg.w_in[i], larg.ch_out[i], larg.ch_out[i], 0 );
	timer.stop();
	latency[y++] = timer.duration();
	timer.start();
	merge_channels<int8_t>(
		(int8_t*)hw_fmap_dfp[3], (int8_t*)hw_fmap_dfp[0], (int8_t*)hw_fmap_dfp[1], larg.h_in[i], larg.w_in[i], 2*larg.ch_out[i] , 1000-2*larg.ch_out[i], 2*larg.ch_out[i] - (1000-2*larg.ch_out[i]) );
	timer.stop();
	latency[y++] = timer.duration();

	// Convert C10 result to float
	timer.start();
	for(int k=0; k < (larg.h_in[i]*larg.w_in[i]*1000); k++ )
		hw_fmap_flp[k] = (float) ( hw_fmap_dfp[1][k/FM_PACK].data[k%FM_PACK] * exp2(-eo[ii]) );
	timer.stop();
	latency[y++] = timer.duration();

	#ifdef USEDEBUG
		check_iresults<float>(
			hw_fmap_flp, SQN_FMAP_MAX_SZ, 0,
			1, 1, 1000, implementation + "CONV10",
			"14_conv10.bin", "data/dfixed/sqn/14_conv10.bin",
			error_sz, error_flag );
	#endif

	// Avgpool
	timer.start();
	sw_avgpool_flp(
		hw_fmap_flp, hw_fmap_flp_o, larg.h_in[i], larg.w_in[i], 1000, 0, 14, 1 );
	timer.stop();
	latency[y++] = timer.duration();

	// Softmax
	timer.start();
	softmax(hw_fmap_flp_o, 1000);
	timer.stop();
	latency[y++] = timer.duration();

	// Print propabilities
	DEBUG( "SqN v1.1 fixed-point HW implementation - Top-5 results:" );
	DEBUG( "-------------------------------------------------------" );
#ifdef USEDEBUG
#if defined(__SIM__)
	print_probs("class_labels.txt", hw_fmap_flp_o, 1000);
#else
	print_probs("data/class_labels.txt", hw_fmap_flp_o, 1000);
#endif
#endif

	// Print latencies
	double total = 0;
	for( int n=0; n<(1+SQN_LN+13); n++)
	{
		DEBUG( "layer " << std::right << std::setw(2) << larg.layer_name_hw[n] << "\t latency (sec): " << latency[n] );
		total += latency[n];
	}
	DEBUG( "TOTAL latency (sec): " << total << std::endl );


	// Free memory
	delete[] hw_fmap_flp;

}
