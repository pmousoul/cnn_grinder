#ifndef SQJ2_TB_HPP
#define SQJ2_TB_HPP


// Include files
#include <iostream>  // for std::cout
#include <fstream>   // for std::ifstream
#include <cstdint>   // for constant length (u)int types
#include <string>    // for string support
#include <cstring>   // for memcpy() 
#include <cmath>     // for calculating exp() softmax layer
#include <algorithm> // for finding max array element in softmax
#include <iomanip>   // for setprecision() - fixed data representation
#include <vector>
#include <numeric>   // for std::iota
#include <typeinfo>  // for using typeid()
#include <unistd.h>  // for using sleep()

#if defined (__SDSCC__)
	#include <stdlib.h>
	#include "sds_lib.h"
	#define malloc(x) (sds_alloc(x))
	#define free(x) (sds_free(x))
#else
	#include <chrono> // For time measurements
#endif

#include "debug.hpp"
#include "timer.hpp"
#include "layer_args.hpp"


#ifdef TRACE
	#include <unistd.h>
	#define TRACE_TIME 1
#endif


// Function declarations and template definitions

// Template used to check inter-layer results
template <typename T>
void check_rslt(
	T* rslt1, T* rslt2, unsigned el_num, std::string layer_name, float error_sz, bool* error_flag )
{
	for ( unsigned i = 0; i < el_num; i++ )
	{
		if ( std::abs( rslt1[i] - rslt2[i] ) < error_sz )
		{
			continue;
		}
		else
		{
			std::cout << "ERROR in "<< layer_name << " layer" << std::endl;
			std::cout << "The values are:" << std::endl;
			std::cout << "value index = " << i << std::endl;
			std::cout << "C/C++ value = " << rslt1[i] << std::endl;
			std::cout << "Caffe/Ristretto value = " << rslt2[i] << std::endl;
			std::cout << std::endl;
			*error_flag = true;
			break;
		}
	}
}

// Template to read number of elements from binary file
template <typename T>
void read_bin_file(
	std::string file_name, T* data_p, unsigned read_el_num, unsigned offset )
{
	// open binary file
	std::ifstream bin_file(file_name, std::ios::in | std::ios::binary);

	// set offset position
	bin_file.seekg( offset*sizeof(T) );

	// read data from file
	bin_file.read(reinterpret_cast<char*>(data_p), read_el_num*sizeof(T));
	bin_file.close();

	if(bin_file)
		;
	else
	{
		DEBUG( "ERROR" << std::endl
		<< "Could not read " << file_name << " data." << std::endl
		<< "Exiting!");
	}
}

// Check inter-layer results of Caffe/Ristretto against the C/C++ results
// Template to avoid re-writing this code after each CNN layer
template <typename T>
void check_iresults(
	T* data_in, unsigned FMAP_MAX_SZ, int8_t exp_val,
	uint16_t hin, uint16_t win, uint16_t chin, std::string layer_name,
	std::string file_name_short, std::string file_name_long,
	float error_sz, bool* error_flag )
{
	// Allocate memory used to verify inter-layer results
	float* fmap_flp_trace = new float[FMAP_MAX_SZ];
	float* fmap_flp_tmp   = new float[FMAP_MAX_SZ];

	unsigned el_num = hin*win*chin;

	if (typeid(T) == typeid(float)) {
		for(unsigned k=0; k<el_num; k++) fmap_flp_tmp[k] = (float)data_in[k];
	} else {
		for(unsigned k=0; k<el_num; k++) fmap_flp_tmp[k] = ( (float)data_in[k] ) * exp2(exp_val);
	}

	#if defined(__SIM__)
		read_bin_file<float>( file_name_short, (float *)fmap_flp_trace, el_num, 0 );
	#else
		read_bin_file<float>( file_name_long, (float *)fmap_flp_trace, el_num, 0 );
	#endif
	check_rslt<float>(fmap_flp_tmp, fmap_flp_trace, el_num, layer_name, error_sz, error_flag );

	// Free memory
	delete[] fmap_flp_trace;
	delete[] fmap_flp_tmp;
}

// Template to reshape the input 
template <typename T>
void reshape_input(
	T* data_p_in, T* data_p_out,
	uint16_t h_in, uint16_t w_in, uint16_t ch_in, uint16_t ch_out,
	uint8_t kernel, uint8_t stride, uint8_t pad )
{
	// calculate output dimensions
	uint16_t h_out = ( (h_in - kernel + 2*pad) / stride ) + 1;
	uint16_t w_out = ( (w_in - kernel + 2*pad) / stride ) + 1;

	// Reshape
	uint16_t WIxCHI = w_in*ch_in;
	T* out_tmp = data_p_out;
	T* in_tmp  = data_p_in;
	for ( uint16_t h=0; h<h_out; h++ )
	{
		for ( uint16_t w=0; w<w_out; w++ )
		{
			// calculate the image window
			int16_t wh_s = h*stride - pad;
			int16_t wh_e = h*stride - pad + kernel;
			int16_t ww_s = w*stride - pad;
			int16_t ww_e = w*stride - pad + kernel;

			for ( int16_t wh = wh_s; wh < wh_e; wh++ )
			{
				for ( int16_t ww = ww_s; ww < ww_e; ww++ )
				{
					for ( uint8_t chi = 0; chi < ch_in; chi++ )
					{
						if ( (wh < 0) || (ww < 0) || (wh > (h_in-1)) || (ww > (w_in-1)) )
						{
							*(out_tmp++) = 0;
						}
						else
						{
							*(out_tmp++) = in_tmp[wh*WIxCHI + ww*ch_in + chi];
						}
					}
				}
			}

			uint8_t zero_pad = ch_out - ( kernel*kernel*ch_in );
			for ( uint8_t i=0; i<zero_pad; i++)
			{
				*(out_tmp++) = 0;
			}
		}
	}
}

// Template to merge channels
template <typename T>
void merge_channels(
	T* data_p_in1, T* data_p_in2, T* data_p_out,
	uint8_t h_num, uint8_t w_num, uint16_t ch_num1, uint16_t ch_num2, uint16_t padding )
{
	T *in1, *in2, *out;
	in1 = data_p_in1;
	in2 = data_p_in2;
	out = data_p_out;

	unsigned HxW = (unsigned)h_num*(unsigned)w_num;
	for(unsigned m=0; m<HxW; m++)
	{
		memcpy( out, in1, ch_num1*sizeof(T) );
		out+=ch_num1;
		in1+=ch_num1;
		memcpy( out, in2, ch_num2*sizeof(T) );
		out+=ch_num2;
		in2+=ch_num2 + padding;
	}
}

// Template to write number of elements to binary file
template <typename T>
void write_bin_file(
	std::string file_name, T* data_p, unsigned write_el_num )
{
	// open binary file
	std::ofstream bin_file(file_name, std::ios::out | std::ios::binary);

	// read data from file
	bin_file.write(reinterpret_cast<char*>(data_p), write_el_num*sizeof(T));
	bin_file.close();

	if(bin_file)
		;
	else
	{
		DEBUG( "ERROR" << std::endl
		<< "Could not write data to " << file_name << "." << std::endl
		<< "Exiting!");
	}
}

void sw_conv_flp(
	float *fmap_in, float *params, float *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in, uint16_t ch_out,
	uint8_t pad, uint8_t kernel, uint8_t stride, uint8_t use_relu );

void sw_conv_dfp(
	int8_t *fmap_in, int8_t *params, int8_t *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
	int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu );

void sw_conv_fip(
	int16_t *fmap_in, int8_t *params, int16_t *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
	int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu );

// Template floating/fixed-point implementation of the max-pool layer
// Zero-pading the input is done on the fly
template <typename T>
void sw_maxpool(
	T *fmap_in, T *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint8_t ppad, uint8_t pkernel, uint8_t pstride )
{
	// Calculate output fmap dimensions
	uint8_t HO = ( (h_in - pkernel + ppad) / pstride ) + 1;
	uint8_t WO = ( (w_in - pkernel + ppad) / pstride ) + 1;

	// Perform max-pooling
	uint16_t CH = ch_in;
	uint16_t WOxCH = WO   * CH;
	uint16_t WIxCH = w_in * CH;

	for (uint8_t ho=0; ho<HO; ho++)
	{
		for (uint8_t wo=0; wo<WO; wo++)
		{
			// input window
			uint8_t wh_s = ho*pstride;
			uint8_t ww_s = wo*pstride;
			uint8_t wh_e = ho*pstride + pkernel;
			uint8_t ww_e = wo*pstride + pkernel;

			for (uint8_t wh=wh_s; wh<wh_e; wh++)
			{
				for (uint8_t ww=ww_s; ww<ww_e; ww++)
				{
					for (uint16_t ch=0; ch<CH; ch++)
					{
						// output index
						unsigned idx = ho*WOxCH + wo*CH + ch;

						// we might have to zero-pad the input
						T tmp = ( ( wh > (h_in-1) ) || ( ww > (w_in-1) ) ) ? 0 : fmap_in[wh*WIxCH + ww*CH + ch]; 

						if ( ( wh==wh_s ) && ( ww==ww_s ) )
							fmap_out[idx] = tmp;
						else
							fmap_out[idx] =
								( fmap_out[idx] < tmp ) ? tmp : fmap_out[idx];
					}
				}
			}
		}
	}
}

void sw_avgpool_flp(
	float *fmap_in, float *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint8_t ppad, uint8_t pkernel, uint8_t pstride );

void softmax(float *fmap, uint16_t size);

std::fstream& goto_line(std::fstream& file, unsigned num);

// Template used for sorting the indexes of vector's elements
template <typename T>
std::vector<std::size_t> sort_indexes(const std::vector<T> &v)
{

	// initialize original index locations
	std::vector<std::size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),
	[&v](std::size_t i1, std::size_t i2) {return v[i1] < v[i2];});

  return idx;
}

void print_probs(
	std::string class_labels, float* fmap, uint16_t size );

// SqN v1.1 SW Floating-point implementation
void sw_sqn_flp(
	float* sw_fmap_flp[4], float* sw_params_flp[SQN_LN], Largs_sqn& larg, float error_sz, bool* error_flag );

// ZynqNet SW Floating-point implementation
void sw_zqn_flp(
	float* sw_fmap_flp[4], float* sw_params_flp[ZQN_LN], float* sw_fmap_flp_o, Largs_zqn& larg, float error_sz, bool* error_flag );

// SqN v1.1 SW 8-bit dynamic fixed-point implementation
void sw_sqn_dfp(
	int8_t* sw_fmap_dfp[4], int8_t* sw_params_dfp[SQN_LN], float* sw_fmap_flp_o, Largs_sqn& larg, float error_sz, bool* error_flag );

// SqN v1.1 SW Fixed-point implementation
void sw_sqn_fip(
	int16_t* sw_fmap_dfp[4], int8_t* sw_params_dfp[SQN_LN], float* sw_fmap_flp_o, Largs_sqn& larg, float error_sz, bool* error_flag );

// ZynqNet SW Fixed-point implementation
void sw_zqn_dfp(
	int8_t* sw_fmap_dfp[4], int8_t* sw_params_dfp[ZQN_LN], float* sw_fmap_flp_o, Largs_zqn& larg, float error_sz, bool* error_flag );

// Test SqueezeNet v1.1 8-bit dynamic fixed-point implementation
int test_sqn_dfp(float error_sz);

// Test SqueezeNet v1.1 fixed-point implementation
int test_sqn_fip(float error_sz);

// Test SqueezeNet v1.1 fixed-point separated HW accelerator implementation
int test_sqn_fips(float error_sz);

// Test ZynqNet 8-bit dynamic fixed-point implementation
int test_zqn_dfp(float error_sz);

// Test ZynqNet floating-point implementation
int test_zqn_flp(float error_sz);

#endif
