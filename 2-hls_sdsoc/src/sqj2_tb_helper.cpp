// Testbench helper functions

#include "sqj2_tb.hpp"
#include <math.h>


// Floating-point cnn 3d convolution software implementation
// with zero padding on the fly
void sw_conv_flp(
	float *fmap_in, float *params, float *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride, uint8_t use_relu )
{

	// pre-calculate terms used in the loops
	uint8_t h_out    = ( (h_in - kernel + 2*pad) / stride ) + 1;
	uint8_t w_out    = ( (w_in - kernel + 2*pad) / stride ) + 1;
	uint16_t WIxCHI  = w_in*ch_in;

	// Introduce a temporary pointer that will be modified
	float *_fmaps_out = fmap_out;

	for ( uint8_t ho = 0; ho < h_out; ho++ )
	{
		for ( uint8_t wo = 0; wo < w_out; wo++ )
		{
			// Set the pointers to the beggining of weights, bias 
			float *_weights = params;
			float *_bias = params + ch_out*kernel*kernel*ch_in;

			// calculate the image window
			int16_t wh_s = ho*stride - pad;
			int16_t wh_e = ho*stride - pad + kernel;
			int16_t ww_s = wo*stride - pad;
			int16_t ww_e = wo*stride - pad + kernel;

			for ( uint16_t cho = 0; cho < ch_out; cho++)
			{
				// initialize result with bias
				float result = *( _bias++ ); // intermediate result

				for ( int16_t wh = wh_s; wh < wh_e; wh++ )
				{
					for ( int16_t ww = ww_s; ww < ww_e; ww++ )
					{
						for ( uint16_t chi = 0; chi < ch_in; chi++ )
						{
							if ( (wh < 0) || (ww < 0) || (wh > (h_in-1)) || (ww > (w_in-1)) )
							{
								_weights++;
							}
							else
							{
								result +=
									fmap_in[wh * WIxCHI + ww * ch_in + chi] * *( _weights++ );
							}
						}
					}
				}
				// use ReLU as activation function
				*( _fmaps_out++ ) = ( use_relu && ( result < 0 ) ) ? 0 : result;
			}
		}
	}
}


// 8-bit dynamic fixed-point cnn 3d convolution software implementation
// with zero padding on the fly
void sw_conv_dfp(
	int8_t *fmap_in, int8_t *params, int8_t *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
	int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu )
{

	// pre-calculate terms used in the loops
	uint8_t h_out    = ( (h_in - kernel + 2*pad) / stride ) + 1;
	uint8_t w_out    = ( (w_in - kernel + 2*pad) / stride ) + 1;
	uint16_t WIxCHI  = w_in*ch_in;

	// Introduce a temporary pointer that will be modified
	int8_t *_fmaps_out = fmap_out;

	for ( uint8_t ho = 0; ho < h_out; ho++ )
	{
		for ( uint8_t wo = 0; wo < w_out; wo++ )
		{
			// Set the pointers to the beggining of weights, bias 
			int8_t *_weights = params;
			int8_t *_bias = params + ch_out*kernel*kernel*ch_in;

			// calculate the image window
			int16_t wh_s = ho*stride - pad;
			int16_t wh_e = ho*stride - pad + kernel;
			int16_t ww_s = wo*stride - pad;
			int16_t ww_e = wo*stride - pad + kernel;

			for ( uint16_t cho = 0; cho < ch_out; cho++)
			{
				// initialize result with zero
				int32_t result = 0;

				for ( int16_t wh = wh_s; wh < wh_e; wh++ )
				{
					for ( int16_t ww = ww_s; ww < ww_e; ww++ )
					{
						for ( uint16_t chi = 0; chi < ch_in; chi++ )
						{
							if ( (wh < 0) || (ww < 0) || (wh > (h_in-1)) || (ww > (w_in-1)) )
							{
								_weights++;
							}
							else
							{
								result +=
									fmap_in[wh * WIxCHI + ww * ch_in + chi] * *( _weights++ );
							}
						}
					}
				}

				float temp = (float)result * exp2(eo-ep-ei);
				temp += *( _bias++ ) * exp2(eo-ep);
				temp = round(temp);

				temp = (temp > 127)  ? 127 : temp; // Overflow
				temp = (temp < -128) ? -128 : temp; // Underflow

				result = (int8_t) temp;

				// use ReLU as activation function
				result = ( use_relu && (result < 0) ) ? 0 : result;

				*( _fmaps_out++ ) = result;
			}
		}
	}
}


// Fixed-point cnn 3d convolution software implementation
// with zero padding on the fly
void sw_conv_fip(
	int16_t *fmap_in, int8_t *params, int16_t *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
	int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu )
{

	// pre-calculate terms used in the loops
	uint8_t h_out    = ( (h_in - kernel + 2*pad) / stride ) + 1;
	uint8_t w_out    = ( (w_in - kernel + 2*pad) / stride ) + 1;
	uint16_t WIxCHI  = w_in*ch_in;

	// Introduce a temporary pointer that will be modified
	int16_t *_fmaps_out = fmap_out;

	for ( uint8_t ho = 0; ho < h_out; ho++ )
	{
		for ( uint8_t wo = 0; wo < w_out; wo++ )
		{
			// Set the pointers to the beggining of weights, bias 
			int8_t *_weights = params;
			int8_t *_bias = params + ch_out*kernel*kernel*ch_in;

			// calculate the image window
			int16_t wh_s = ho*stride - pad;
			int16_t wh_e = ho*stride - pad + kernel;
			int16_t ww_s = wo*stride - pad;
			int16_t ww_e = wo*stride - pad + kernel;

			for ( uint16_t cho = 0; cho < ch_out; cho++)
			{
				// initialize result with bias
				int32_t result = 0;
				if ( ei > 0 )
					result = result << ei;
				else
					result = result >> (-ei);

				for ( int16_t wh = wh_s; wh < wh_e; wh++ )
				{
					for ( int16_t ww = ww_s; ww < ww_e; ww++ )
					{
						for ( uint16_t chi = 0; chi < ch_in; chi++ )
						{
							if ( (wh < 0) || (ww < 0) || (wh > (h_in-1)) || (ww > (w_in-1)) )
							{
								_weights++;
							}
							else
							{
								result +=
									fmap_in[wh * WIxCHI + ww * ch_in + chi] * *( _weights++ );
							}
						}
					}
				}
				float temp = (float)result * exp2(eo-ep-ei);
				temp += *( _bias++ ) * exp2(eo-ep);
				temp = round(temp);

				temp = (temp > 32767)  ? 32767 : temp;    // Overflow
				temp = (temp < -32768) ? -32768 : temp; // Underflow

				result = (int16_t) temp;

				// use ReLU as activation function
				result = ( use_relu && (result < 0) ) ? 0 : result;

				*( _fmaps_out++ ) = result;
			}
		}
	}
}


// Floating-point implementation of the avg-pool layer
// Zero-pading the input is done on the fly
void sw_avgpool_flp(
	float *fmap_in, float *fmap_out,
	uint8_t h_in, uint8_t w_in, uint16_t ch_in,
	uint8_t ppad, uint8_t pkernel, uint8_t pstride )
{
	// Calculate output fmap dimensions
	uint8_t HO = ( (h_in - pkernel + ppad) / pstride ) + 1;
	uint8_t WO = ( (w_in - pkernel + ppad) / pstride ) + 1;

	// Perform avg-pooling
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
						float tmp = ( ( wh > (h_in-1) ) || ( ww > (w_in-1) ) ) ? 0 : fmap_in[wh*WIxCH + ww*CH + ch]; 

						if ( ( wh==wh_s ) && ( ww==ww_s ) )
							fmap_out[idx] = tmp;
						else
							fmap_out[idx] += tmp;

						if ( ( wh==wh_e-1 ) && ( ww==ww_e-1 ) )
							fmap_out[idx] /= (pkernel*pkernel);
					}
				}
			}
		}
	}
}


// Floating-point version of softmax layer
void softmax(
	float *fmap, uint16_t size )
{
	// find max element
	float max_el = *std::max_element(fmap, fmap + size );

	// keep in a vector the value exp(fmap[i] - max_el)
	std::vector<float> vec;
	for (uint16_t i=0; i<size; i++)
	{
		vec.push_back(exp(fmap[i] - max_el));
	}

	// find the sum of the vec elements
	float vec_sum = 0;
	for (auto& n : vec) vec_sum += n;

	for (uint16_t i=0; i<size; i++)
	{
		fmap[i] = vec[i]/vec_sum;
	}

}


// Go to specific file line
std::fstream& goto_line(
	std::fstream& file, unsigned num )
{
	file.seekg(std::ios::beg);

	for(unsigned i=0; i < num - 1; ++i)
	{
		file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
	}
	return file;
}


// Print the "propabilities" calculated by softmax
void print_probs(std::string class_labels, float* fmap, uint16_t size)
{
	std::fstream file( class_labels );

	std::vector<float> tmp(fmap, fmap + size );

	std::vector<size_t> idx = sort_indexes(tmp);
	std::cout << "\tClass number:\t" << "Probability:\t" << "Class name:" << std::endl;
	std::cout << "\t-------------\t" << "------------\t" << "-----------" << std::endl;

	for(unsigned i=tmp.size()-1; i>tmp.size()-6; i--)
	{
		// represent the classes from [1,1000]
		int il = idx[i]+1;

		goto_line(file, il);
		std::string line;
		getline(file, line);

		std::cout << std::fixed << std::setprecision(7) << std::setfill('0');
		std::cout << "\t" << il << "\t\t" << std::setw(7) << tmp[idx[i]] << "\t" << line << std::endl;
	}
	std::cout << std::endl;
}
