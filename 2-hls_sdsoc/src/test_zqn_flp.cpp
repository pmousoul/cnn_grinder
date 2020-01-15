// Test ZynqNet implementation

#include "sqj2_tb.hpp"
#include "hw_func_flp.hpp"


int test_zqn_flp(float error_sz)
{
	// Error flag to indicate error
	// when comparing inter-layer results
	// against caffe/ristretto implementations
	bool error_flag = false;

	// Load layer arguments
	Largs_zqn larg;


	// SW Floating-point implementation
	//*********************************

	// Allocate memory for layer io
	float* sw_fmap_flp[4];
	for ( int i=0; i < 4; i++ ) sw_fmap_flp[i] = new float[ZQN_FMAP_MAX_SZ];
	float* sw_fmap_flp_o = new float[ZQN_FMAP_MAX_SZ];

	// Read input image
	#if defined(__SIM__)
		read_bin_file<float>( "image_float.bin", sw_fmap_flp[0], ZQN_IMG_SZ, 0 );
	#else
		read_bin_file<float>( "data/float/zqn/image_float.bin", sw_fmap_flp[0], ZQN_IMG_SZ, 0 );
	#endif

	// Allocate memory for parameters and load the parameters
	float* sw_params_flp[ZQN_LN];
	unsigned offset = 0;
	unsigned size = 0;
	for ( int i=0; i < ZQN_LN; i++ )
	{
		offset += size;
		size = ( larg.ch_out[i] * larg.kernel[i] * larg.kernel[i] * larg.ch_in[i] ) + larg.ch_out[i];
		sw_params_flp[i] = new float[ size ];
		#if defined(__SIM__)
			read_bin_file<float>( "params_float.bin", sw_params_flp[i], size, offset );
		#else
			read_bin_file<float>( "data/float/zqn/params_float.bin", sw_params_flp[i], size, offset );
		#endif
	}

	// Run the CNN
	sw_zqn_flp( sw_fmap_flp, sw_params_flp, sw_fmap_flp_o, larg, error_sz, &error_flag );

	// Free memory
	for ( int i=0; i < 4; i++ )  delete[] sw_fmap_flp[i];
	for ( int i=0; i < ZQN_LN; i++ ) delete[] sw_params_flp[i];
	//*********************************


	// HW Floating-point implementation
	//******************************

	// Allocate memory for layer io
	p_fmap_t* hw_fmap_flp[4];
	for ( int i=0; i < 4; i++ ) hw_fmap_flp[i] = (p_fmap_t *)malloc(FMAPI_FIFO_DEPTH * sizeof(p_fmap_t));
	float* hw_fmap_flp_o = new float[ZQN_FMAP_MAX_SZ];

	// Read input image
	#if defined(__SIM__)
		read_bin_file<float>( "image_float.bin", (float *)hw_fmap_flp[0], ZQN_IMG_SZ, 0 );
	#else
		read_bin_file<float>( "data/float/zqn/image_float.bin", (float *)hw_fmap_flp[0], ZQN_IMG_SZ, 0 );
	#endif

	// Allocate memory for parameters and load the parameters
	p_param_t* hw_params_flp[ZQN_LN];
	offset = 0;
	size = 0;
	for ( int i=0; i < ZQN_LN; i++ )
	{
		offset += size;
		size = ( larg.ch_out[i] * larg.kernel[i] * larg.kernel[i] * larg.ch_in[i] ) + larg.ch_out[i];
		#if defined(USE_AFI)
			#if defined(ULTRA96)
				hw_params_flp[i] = (p_param_t *)malloc( ( size/WB_PACK ) * sizeof(p_param_t));
			#else
				hw_params_flp[i] = (p_param_t *)sds_alloc_non_cacheable( ( size/WB_PACK ) * sizeof(p_param_t));
			#endif
		#else
			hw_params_flp[i] = (p_param_t *)malloc( ( size/WB_PACK ) * sizeof(p_param_t));
		#endif
		#if defined(__SIM__)
			read_bin_file<float>( "params_float.bin", (float *)hw_params_flp[i], size, offset );
		#else
			read_bin_file<float>( "data/float/zqn/params_float.bin", (float *)hw_params_flp[i], size, offset );
		#endif
	}

	// Run the CNN
	hw_zqn_flp( hw_fmap_flp, hw_params_flp, hw_fmap_flp_o, larg, error_sz, &error_flag );

	// Free memory
	for ( int i=0; i < 4; i++ )  free( hw_fmap_flp[i] );
	for ( int i=0; i < ZQN_LN; i++ ) free( hw_params_flp[i] );
	//******************************

	// Free memory
	delete[] sw_fmap_flp_o;
	delete[] hw_fmap_flp_o;


	// Return from main in case of inconsistency
	int ret_val;
	if ( error_flag )
	{
		std::cout << std::endl << "Consistency check between Caffe/Ristretto and C/C++ results failed." << std::endl;
		std::cout << "Max absolute error is greater than: " << error_sz << std::endl;
		ret_val = -1;
		return ret_val;
	}
	else
	{
		std::cout << std::endl << "Consistency check between Caffe/Ristretto and C/C++ results passed." << std::endl;
		std::cout << "Max absolute error is less than: " << error_sz << std::endl;
		ret_val = 0;
		return ret_val;
	}

}
