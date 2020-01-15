// HW CNN Convolution implementation

#include "hw_func_fips.hpp"


// Pre-calculate terms

#ifdef ABS
#undef ABS
#endif
#define ABS(n) ((n < 0) ? -n : n)

static float float_mul_pow2(float x, int8_t n)
{
//#pragma AP inline // Always inline this function
DO_PRAGMA ( HLS inline )
   float_num_t x_num, prod;

   x_num.fp_num = x;
#ifndef AESL_FP_MATH_NO_BOUNDS_TESTS
   if (x_num.bexp == 0xFF || x_num.bexp == 0) // pass through NaN, INF & denorm
      prod.fp_num = x_num.fp_num;
   else if (n >= 0 && x_num.bexp >= 255 - n) { // detect and handle overflow
      prod.sign = x_num.sign; //
      prod.bexp = 0xFF;       // +/-INF
      prod.mant = 0;          //
   } else if (n < 0 && x_num.bexp <= ABS(n)) { // handle underflow (doesn't gen denorms)
      prod.sign = x_num.sign; //
      prod.bexp = 0;          // +/-ZERO
      prod.mant = 0;          //
   } else
#endif // AESL_FP_MATH_NO_BOUNDS_TESTS not defined
   {
      prod.sign = x_num.sign;
      prod.bexp = x_num.bexp + n;
      prod.mant = x_num.mant;
   }
   return prod.fp_num;
}

void precalc_terms(
	uint8_t h_in, uint8_t w_in, uint16_t ch_in, uint16_t ch_out,
	uint8_t pad, uint8_t kernel, uint8_t stride, uint8_t *h_out,
	uint8_t *w_out, uint16_t *WIxCHI, uint16_t *KxCHI, uint16_t *KxKxCHI,
	uint16_t *Q_CHO, uint16_t *SxCHI, uint16_t *PADxCHI,
	uint32_t *CHOxKxKxCHI, uint32_t *Q_CHOxKxKxCHI )
{
	*h_out         = ( (h_in - kernel + (pad<<1))>>(stride-1) ) + 1;
	*w_out         = ( (w_in - kernel + (pad<<1))>>(stride-1) ) + 1;
	*WIxCHI        = w_in   * ch_in;
	*KxCHI         = kernel * ch_in;
	*KxKxCHI       = kernel * *KxCHI;
	*Q_CHO         = ch_out >> BIT_SHIFT;
	*SxCHI         = stride * ch_in;
	*PADxCHI       = pad * ch_in;
	*CHOxKxKxCHI   = ch_out * *KxKxCHI;
	*Q_CHOxKxKxCHI = *CHOxKxKxCHI  >> BIT_SHIFT;
}


// Initialize inputs
static void clear_lb(
	fmap_t linebuf[WIxCHI_MAX] )
{
	L_CLEAR_LB_N: for( uint16_t n=0; n != WIxCHI_MAX; n+=CHI_NUM )
	{
		DO_PRAGMA ( HLS pipeline II=1 )

		L_CLEAR_LB_NN: for( uint8_t nn=0; nn != CHI_NUM; nn++ )
		{
			linebuf[n + nn] = 0;
		}
	}
}

static void fill_linebuf(
	p_fmap_t *fmap_in, uint32_t *iidx, fmap_t linebuf[K_MAX][WIxCHI_MAX],
	uint8_t line_num, uint16_t line_sz, uint16_t PADxCHI )
{
	p_fmap_t pfm;

	L_FILL_LB_N: for( uint16_t n=0; n != line_sz; n+=FM_PACK )
	{
		DO_PRAGMA ( HLS loop_tripcount min=WIxCHI_MIN_FMP max=WIxCHI_MAX_FMP )
		DO_PRAGMA ( HLS pipeline II=1 )

		pfm = fmap_in[ (*iidx)++ ];
		uint16_t n_tmp = n / FM_PACK;

		L_FILL_LB_NN: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
			linebuf[line_num][ PADxCHI + n_tmp*FM_PACK + nn ] = pfm.data[ nn ];
	}
}

static void init_linebufs(
	p_fmap_t *fmap_in, uint32_t *iidx, fmap_t linebuf[K_MAX][WIxCHI_MAX],
	uint8_t linebuf_idx[K_MAX], uint16_t line_sz, uint8_t kernel, uint16_t PADxCHI )
{
	if ( kernel == 3 )
	{
		// Init linebuf indices
		linebuf_idx[ 0 ] = static_cast <uint8_t> ( 0 );
		linebuf_idx[ 1 ] = static_cast <uint8_t> ( 1 );
		linebuf_idx[ 2 ] = static_cast <uint8_t> ( 2 );

		if ( PADxCHI == 0 ) // pad==0
		{
			fill_linebuf( fmap_in, iidx, linebuf, 1, line_sz, 0 );
			fill_linebuf( fmap_in, iidx, linebuf, 2, line_sz, 0 );
		}
		else  // pad!=0
		{
			// Clear the linebuffers
			clear_lb( linebuf[ 0 ] );
			clear_lb( linebuf[ 1 ] );
			clear_lb( linebuf[ 2 ] );

			fill_linebuf( fmap_in, iidx, linebuf, 2, line_sz, PADxCHI );
		}
	}
	else // kernel==1
	{
		// Init linebuf index
		linebuf_idx[ 0 ] = static_cast <uint8_t> ( 0 );
	}
}

static void init_params(
	p_param_t *param, uint32_t Q_CHOxKxKxCHI, uint16_t Q_CHO,
	param_t _weights[PAR_FACT][Q_CHOxKxKxCHI_MAX],
	param_t _bias[PAR_FACT][Q_CHO_MAX] )
{
	uint32_t pidx = 0;
	p_param_t ppar;

	// Load weights into PAR_FACT number of weight caches
	L_LOAD_W_I: for ( uint8_t i = 0; i != PAR_FACT; i++ )
	{
		DO_PRAGMA ( HLS unroll )

		L_LOAD_W_N: for ( uint32_t n = 0; n != Q_CHOxKxKxCHI; n+=WB_PACK )
		{
			DO_PRAGMA ( HLS loop_tripcount min=Q_CHOxKxKxCHI_MIN_WBP max=Q_CHOxKxKxCHI_MAX_WBP )
			DO_PRAGMA ( HLS pipeline II=1 )

			ppar = param[ pidx++ ];
			uint32_t n_tmp = n / WB_PACK;

			L_LOAD_W_NN: for ( uint8_t nn = 0; nn != WB_PACK; nn++)
				_weights[i][ ( n_tmp*WB_PACK + nn )] = ppar.data[ nn ];
		}
	}

	// Load bias into temporary bias cache
	// used for fast data streaming into the BRAMs
	// because all param (weights & bias) are packed
	param_t _bias_tmp[CHO_MAX];
	DO_PRAGMA ( HLS array_partition variable=_bias_tmp cyclic factor=WB_PACK dim=1 )
	DO_PRAGMA ( HLS RESOURCE variable=_bias_tmp core=RAM_2P_LUTRAM )

	L_LOAD_BT_N: for ( uint16_t n = 0; n != ( Q_CHO << BIT_SHIFT); n+=WB_PACK )
	{
		DO_PRAGMA ( HLS loop_tripcount min=CHO_MIN_WBP max=CHO_MAX_WBP )
		DO_PRAGMA ( HLS pipeline II=1 )

		ppar = param[ pidx++ ];
		uint16_t n_tmp = n / WB_PACK;

		L_LOAD_BT_NN: for ( uint8_t nn = 0; nn != WB_PACK; nn++)
			_bias_tmp[ ( n_tmp*WB_PACK + nn )] = ppar.data[ nn ];
	}

	// Load bias into PAR_FACT bia caches
	uint16_t bidx = 0;
	L_LOAD_B_I: for ( uint8_t i = 0; i != PAR_FACT; i++ )
	{
		DO_PRAGMA ( HLS unroll )

		L_LOAD_B_N: for ( uint16_t n = 0; n != Q_CHO; n++ )
		{
			DO_PRAGMA ( HLS loop_tripcount min=Q_CHO_MIN max=Q_CHO_MAX )
			DO_PRAGMA ( HLS pipeline II=1 )

			_bias[i][n] = _bias_tmp[bidx];
			bidx++;
		}
	}
}

static void init_caches(
	p_fmap_t *fmap_in, uint32_t *iidx, fmap_t linebuf[K_MAX][WIxCHI_MAX],
	uint8_t linebuf_idx[K_MAX], uint16_t line_sz, uint8_t kernel,
	p_param_t *param, uint32_t Q_CHOxKxKxCHI, uint16_t Q_CHO,
	param_t _weights[PAR_FACT][Q_CHOxKxKxCHI_MAX],
	param_t _bias[PAR_FACT][Q_CHO_MAX], uint16_t PADxCHI )
{
	// initialize line buffer
	init_linebufs( fmap_in, iidx, linebuf, linebuf_idx, line_sz, kernel, PADxCHI );

	// initialize weights and bias caches
	init_params( param, Q_CHOxKxKxCHI, Q_CHO, _weights, _bias );
}


// Shift image tiles "down" in the image
static void shift_index( uint8_t window[K_MAX], uint8_t kernel )
{
	if ( kernel == 3 )
	{
		uint8_t tmp_p = window[2];
		window[2] = window[0];
		window[0] = window[1];
		window[1] = tmp_p;
	}
}


// We fill the last linebuffer with KxCHI input fmap values (K 3D pixels)
// We could use the fill_linebuf() function, but if we do we will get
// wrong performance estimation in Vivado HLS because of the different
// max value of the loop_tripcount directive
static void fill_lb_last_row(
	p_fmap_t *fmap_in, uint32_t *iidx, fmap_t linebuf[K_MAX][WIxCHI_MAX],
	uint8_t line_num, uint16_t KxCHI, uint16_t PADxCHI )
{
	p_fmap_t pfm;

	L_fill_lb_last_row_N: for( uint16_t n=0; n != KxCHI; n+=FM_PACK )
	{
		DO_PRAGMA ( HLS loop_tripcount min=KxCHI_MIN_FMP max=KxCHI_MAX_FMP )
		DO_PRAGMA ( HLS pipeline II=1 )

		pfm = fmap_in[ (*iidx)++ ];
		uint16_t n_tmp = n / FM_PACK;

		L_fill_lb_last_row_NN: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
			linebuf[line_num][ PADxCHI + n_tmp*FM_PACK + nn ] = pfm.data[ nn ];
	}
}

static void shift_linebuf(
	p_fmap_t *fmap_in, uint32_t *iidx, fmap_t linebuf[K_MAX][WIxCHI_MAX],
	uint8_t linebuf_idx[K_MAX], uint16_t WIxCHI, uint16_t KxCHI,
	uint8_t kernel, uint8_t stride, uint8_t ho, uint8_t h_out,
	uint16_t PADxCHI )
{
	// When (stride == 2) we need to fill an additional image line into
	// the linebuffer.
	if ( (stride == 2) && ( ho != 0 ) )
	{
		shift_index( linebuf_idx, kernel );
		fill_linebuf( fmap_in, iidx, linebuf, linebuf_idx[ kernel - 1 ], WIxCHI, PADxCHI );
	}

	// KxCHI input fmap values into the linebuffer or if we reached the 
	// last output fmap line pad the next line with zeros
	// In case we have a stride==2 and pad==1, it is not required to clear
	// the last linebuffer line because we reach the required line number
	// at the last line of the input-feature-map (because we insert an
	// additional line during cache initialization)
	shift_index( linebuf_idx, kernel );
	if ( (stride == 1) && (ho == h_out-1) && (PADxCHI != 0) )
	{
		clear_lb( linebuf[ linebuf_idx[ kernel - 1 ] ] );
	}
	else
	{
		fill_lb_last_row( fmap_in, iidx, linebuf, linebuf_idx[ kernel - 1 ], KxCHI, PADxCHI );
	}
}


// Initialize image window
static void init_linebuf_win(
	fmap_t linebuf[K_MAX][WIxCHI_MAX], uint8_t linebuf_idx[K_MAX],
	fmap_t linebuf_win[KxKxCHI_MAX], uint16_t KxCHI, uint16_t KxKxCHI,
	uint16_t pixel_pt )
{
	L_INIT_IM_WIN_CH: for ( uint16_t iw_ch = 0; iw_ch != KxKxCHI; iw_ch+=CHI_NUM )
	{
		DO_PRAGMA ( HLS loop_tripcount min=KxKxCHI_MIN_CHN max=KxKxCHI_MAX_CHN )
		DO_PRAGMA ( HLS pipeline II=1 )

		uint16_t pix_tmp = pixel_pt / CHI_NUM;
		uint16_t iw_ch_tmp1 = iw_ch / CHI_NUM;

		uint8_t idx = ( iw_ch >= KxCHI ) ? 1 : 0;
		idx = ( iw_ch >= ( KxCHI << 1) ) ? 2 : idx;

		uint16_t iw_ch_tmp2 = ( iw_ch-(idx*KxCHI) ) / CHI_NUM;

		L_INIT_IM_WIN_CHI_N: for ( uint8_t n = 0; n != CHI_NUM; n++ )
		{
			linebuf_win[iw_ch_tmp1*CHI_NUM + n] = linebuf[ linebuf_idx[ idx ] ][ (pix_tmp+iw_ch_tmp2)*CHI_NUM + n ];
		}
	}
}


// Calculate output channels
static void macc_dsp(
	fmap_t linebuf_win[CHI_NUM], param_t _weights[CHI_NUM], acc_t *macc_buf )
{
	DO_PRAGMA ( HLS inline )

	mul_t mul_buf[CHI_NUM];

	// CHI_NUM concurrent multiplications
	L_MUL: for ( uint8_t n = 0; n != CHI_NUM; n++ )
	{
		DO_PRAGMA ( HLS pipeline II=1 )
		mul_t tmp;
		fmap_t itmp = linebuf_win[ n ];
		param_t wtmp = _weights[ n ];
		DO_PRAGMA ( HLS resource variable=tmp core=DSP48 )
		tmp =  itmp * wtmp;
		mul_buf[n] = tmp;
	}

	// Adder tree
#if CHI_NUM == 16

	acc_t win1[CHI_NUM/2];
	acc_t win2[CHI_NUM/4];
	acc_t win3[CHI_NUM/8];

	L_MACCP0: for(uint8_t n=0; n<CHI_NUM/2; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win1[n] = mul_buf[n] + mul_buf[CHI_NUM/2+n];
	}
	L_MACCP1: for(uint8_t n=0; n<CHI_NUM/4; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win2[n] = win1[n] +  win1[CHI_NUM/4+n];
	}
	L_MACCP2: for(uint8_t n=0; n<CHI_NUM/8; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win3[n] = win2[n] +  win2[CHI_NUM/8+n];
	}

	acc_t maccp;
	#ifdef TRACE 
		DO_PRAGMA ( HLS resource variable=maccp core=AddSubnS )
	#endif
	maccp = win3[0] + win3[1];

#elif CHI_NUM == 8

	acc_t win1[CHI_NUM/2];
	acc_t win2[CHI_NUM/4];

	L_MACCP0: for(uint8_t n=0; n<CHI_NUM/2; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win1[n] = mul_buf[n] + mul_buf[CHI_NUM/2+n];
	}
	L_MACCP1: for(uint8_t n=0; n<CHI_NUM/4; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win2[n] = win1[n] +  win1[CHI_NUM/4+n];
	}

	acc_t maccp;
	#ifdef TRACE 
		DO_PRAGMA ( HLS resource variable=maccp core=AddSubnS )
	#endif
	maccp = win2[0] + win2[1];

#elif CHI_NUM == 4

	acc_t win1[CHI_NUM/2];

	L_MACCP0: for(uint8_t n=0; n<CHI_NUM/2; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win1[n] = mul_buf[n] + mul_buf[CHI_NUM/2+n];
	}

	acc_t maccp;
	#ifdef TRACE 
		DO_PRAGMA ( HLS resource variable=maccp core=AddSubnS )
	#endif
	maccp = win1[0] + win1[1];

#endif

	*macc_buf +=  maccp;

}

static void macc_lut(
	fmap_t linebuf_win[CHI_NUM], param_t _weights[CHI_NUM], acc_t *macc_buf )
{
	DO_PRAGMA ( HLS inline )

	mul_t mul_buf[CHI_NUM];

	// CHI_NUM concurrent multiplications
	L_MUL: for ( uint8_t n = 0; n != CHI_NUM; n++ )
	{
		DO_PRAGMA ( HLS pipeline II=1 )
		mul_t tmp;
		fmap_t itmp = linebuf_win[ n ];
		param_t wtmp = _weights[ n ];
		DO_PRAGMA ( HLS resource variable=tmp core=Mul_LUT )
		tmp =  itmp * wtmp;
		mul_buf[n] = tmp;
	}

	// Adder tree
#if CHI_NUM == 16

	acc_t win1[CHI_NUM/2];
	acc_t win2[CHI_NUM/4];
	acc_t win3[CHI_NUM/8];

	L_MACCP0: for(uint8_t n=0; n<CHI_NUM/2; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win1[n] = mul_buf[n] + mul_buf[CHI_NUM/2+n];
	}
	L_MACCP1: for(uint8_t n=0; n<CHI_NUM/4; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win2[n] = win1[n] +  win1[CHI_NUM/4+n];
	}
	L_MACCP2: for(uint8_t n=0; n<CHI_NUM/8; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win3[n] = win2[n] +  win2[CHI_NUM/8+n];
	}

	acc_t maccp;
	#ifdef TRACE 
		DO_PRAGMA ( HLS resource variable=maccp core=AddSubnS )
	#endif
	maccp = win3[0] + win3[1];

#elif CHI_NUM == 8

	acc_t win1[CHI_NUM/2];
	acc_t win2[CHI_NUM/4];

	L_MACCP0: for(uint8_t n=0; n<CHI_NUM/2; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win1[n] = mul_buf[n] + mul_buf[CHI_NUM/2+n];
	}
	L_MACCP1: for(uint8_t n=0; n<CHI_NUM/4; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win2[n] = win1[n] +  win1[CHI_NUM/4+n];
	}

	acc_t maccp;
	#ifdef TRACE 
		DO_PRAGMA ( HLS resource variable=maccp core=AddSubnS )
	#endif
	maccp = win2[0] + win2[1];

#elif CHI_NUM == 4

	acc_t win1[CHI_NUM/2];

	L_MACCP0: for(uint8_t n=0; n<CHI_NUM/2; n++)
	{
		DO_PRAGMA ( HLS unroll )
		win1[n] = mul_buf[n] + mul_buf[CHI_NUM/2+n];
	}

	acc_t maccp;
	#ifdef TRACE 
		DO_PRAGMA ( HLS resource variable=maccp core=AddSubnS )
	#endif
	maccp = win1[0] + win1[1];

#endif

	*macc_buf +=  maccp;
}

static void calc_ch_out(
	fmap_t linebuf_win[KxKxCHI_MAX],
	param_t _weights[PAR_FACT][Q_CHOxKxKxCHI_MAX],
	// param_t _bias[PAR_FACT][Q_CHO_MAX],
	float _fmaps_out[PAR_FACT][Q_CHO_MAX],
	uint16_t KxKxCHI, uint32_t Q_CHOxKxKxCHI
	//int8_t ei
	)
{
	DO_PRAGMA ( HLS inline off )

	// array used to replicate the input fmap
	fmap_t linebuf_win_tmp[PAR_FACT][CHI_NUM];
	DO_PRAGMA ( HLS array_partition variable=linebuf_win_tmp complete dim=0 )

	// array used to hold the MACC results
	acc_t macc_buf[PAR_FACT];
	DO_PRAGMA ( HLS array_partition variable=macc_buf complete dim=0 )

	uint16_t q_cho=0;
	uint16_t times_KxKxCHI = KxKxCHI;

	L_INIT_MACCBUF: for ( uint8_t i = 0; i != PAR_FACT; i++ )
	{
		DO_PRAGMA ( HLS unroll )
		macc_buf[i] = static_cast <acc_t> ( 0 );
	}

	L_Q_CHOxKxKxCHI: for ( uint16_t idx=0, idy=0; idx != Q_CHOxKxKxCHI; idx+=CHI_NUM, idy+=CHI_NUM )
	{
		DO_PRAGMA ( HLS loop_tripcount min=Q_CHOxKxKxCHI_MIN_CHN max=Q_CHOxKxKxCHI_MAX_CHN )
		DO_PRAGMA ( HLS pipeline II=1 )

		DO_PRAGMA ( HLS dependence variable=linebuf_win_tmp intra false )
		DO_PRAGMA ( HLS dependence variable=_weights intra false )

		uint32_t idy_t = idy / CHI_NUM;

		// replicate linebuf window for broadcasting to the compute units
		L_REPL_WIN_I: for ( uint8_t i = 0; i != PAR_FACT; i++ )
		{
			L_REPL_WIN_N: for ( uint8_t n = 0; n != CHI_NUM; n++ )
			{
				linebuf_win_tmp[ i ][ n ] = linebuf_win[ idy_t*CHI_NUM + n ];
			}
		}

		L_MACC_DSP: for (uint8_t i=0; i != DSP_FACT; i++)
		{
			DO_PRAGMA ( HLS unroll )

			macc_dsp( linebuf_win_tmp[ i ], &_weights[ i ][ idx ], &macc_buf[ i ] );
		}

		L_MACC_LUT: for (uint8_t i=DSP_FACT; i != PAR_FACT; i++)
		{
			DO_PRAGMA ( HLS unroll )

			macc_lut( linebuf_win_tmp[ i ], &_weights[ i ][ idx ], &macc_buf[ i ] );
		}

		if ( ( idx + CHI_NUM ) == times_KxKxCHI )
		{

			L_WRITE_MACC_RES: for ( uint8_t i = 0; i != PAR_FACT; i++ )
			{
				DO_PRAGMA ( HLS pipeline II=1 )
				DO_PRAGMA ( HLS dependence variable=_fmaps_out inter false )

				// macc_buf[ i ] = macc_buf[ i ] >> ( -(eo-ep-ei) );

				// macc_buf[ i ] = (macc_buf[ i ] > 32767)  ? static_cast <acc_t> (  32767 ) : macc_buf[ i ]; // Saturation
				// macc_buf[ i ] = (macc_buf[ i ] < -32768) ? static_cast <acc_t> ( -32768 ) : macc_buf[ i ]; // Saturation

				_fmaps_out[ i ][ q_cho ] =  static_cast <float> ( macc_buf[ i ] );

				macc_buf[i] = static_cast <acc_t> ( 0 );

			}

			idy = -CHI_NUM;

			times_KxKxCHI += KxKxCHI;

			q_cho++;

		}

	}
 
}

static void write_pix(
	float _fmaps_out[PAR_FACT][Q_CHO_MAX],
	param_t _bias[PAR_FACT][Q_CHO_MAX],
	uint16_t Q_CHO,
	fmap_t *out_pix,
	int8_t ei, int8_t eo, int8_t ep )
{
	uint16_t idx = 0;

	L_WP_I: for ( uint8_t i = 0; i != PAR_FACT; i++ )
	{
		DO_PRAGMA ( HLS pipeline II=1 )

		L_WP_N: for ( uint16_t n = 0; n != Q_CHO; n++ )
		{
			DO_PRAGMA ( HLS loop_tripcount min=Q_CHO_MIN max=Q_CHO_MAX )

			float temp_b = (float)_bias[i][ n ];
			float temp1 = float_mul_pow2(_fmaps_out[i][ n ], eo-ep-ei);
			float temp2 = float_mul_pow2(temp_b, eo-ep);
			float temp = temp1 + temp2;
			temp = round(temp);

			temp = (temp > 32767)  ? 32767 : temp; // Overflow
			temp = (temp < -32768) ? -32768 : temp; // Underflow

			out_pix[ idx++ ] = static_cast<fmap_t>( temp );
		}
	}
}

static void pixel_calc(
	fmap_t linebuf_win[KxKxCHI_MAX],
	param_t _weights[PAR_FACT][Q_CHOxKxKxCHI_MAX],
	param_t _bias[PAR_FACT][Q_CHO_MAX], uint16_t KxKxCHI,
	uint32_t Q_CHOxKxKxCHI, uint16_t Q_CHO, fmap_t *out_pix,
	int8_t ei, int8_t eo, int8_t ep )
{
	DO_PRAGMA ( HLS dataflow )
	DO_PRAGMA ( HLS inline off )

	float _fmaps_out[PAR_FACT][Q_CHO_MAX];
	DO_PRAGMA ( HLS array_partition variable=_fmaps_out complete dim=1 )
	DO_PRAGMA ( HLS stream variable=_fmaps_out )

	// Calculate channels out for one pixel
	calc_ch_out(
		linebuf_win, _weights, _fmaps_out, KxKxCHI, Q_CHOxKxKxCHI );

	// Write back calculated channels
	write_pix( _fmaps_out, _bias, Q_CHO, out_pix, ei, eo, ep );
}


// update linebuf and image window with new pixels
static void fill_lb_pixel(
	p_fmap_t *fmap_in, uint32_t *iidx, fmap_t linebuf[K_MAX][WIxCHI_MAX],
	uint8_t last_linebuf, uint16_t ch_in, uint16_t pixel_p,
	uint16_t PADxCHI, uint8_t ho, uint8_t h_out, uint8_t stride )
{
	if ( !( (stride == 1) && (ho == h_out-1) && (PADxCHI != 0) ) )
	{
		p_fmap_t pfm;

		L_FILL_LB_PIX_N: for( uint16_t n=0; n != ch_in; n+=FM_PACK )
		{
			DO_PRAGMA ( HLS loop_tripcount min=CHI_MIN_FMP max=CHI_MAX_FMP )
			DO_PRAGMA ( HLS pipeline II=1 )

			pfm = fmap_in[ (*iidx)++ ];

			uint16_t index = (pixel_p + n) / FM_PACK;

			L_FILL_LB_PIX_NN: for ( uint8_t nn = 0; nn != FM_PACK; nn++ )
				linebuf[ last_linebuf ] [ PADxCHI + index*FM_PACK + nn ] = pfm.data[ nn ];
		}
	}
}

static void update_linebuf_win(
	p_fmap_t *fmap_in, uint32_t *iidx, fmap_t linebuf[K_MAX][WIxCHI_MAX],
	uint8_t kernel, uint16_t SxCHI, uint16_t WIxCHI,
	uint16_t *lb_pixel_pt, uint8_t linebuf_idx[K_MAX],
	fmap_t linebuf_win[KxKxCHI_MAX], uint16_t KxCHI, uint16_t KxKxCHI,
	uint16_t *pixel_iwp, uint16_t PADxCHI, uint8_t ho, uint8_t h_out, uint8_t stride )
{
	// fill linebuffer with a new pixel if ( wo < w_out-1 )
	// we need to load all pixels in the row because we read the input sequentially
	if ( *lb_pixel_pt + (SxCHI<<1) > WIxCHI ) SxCHI = WIxCHI - *lb_pixel_pt;
	fill_lb_pixel( fmap_in,  iidx, linebuf, linebuf_idx[ kernel - 1 ], SxCHI,
		*lb_pixel_pt, PADxCHI, ho, h_out, stride );
	*lb_pixel_pt += SxCHI;

	// update image window
	if ( *pixel_iwp < WIxCHI )
	{
		init_linebuf_win( linebuf, linebuf_idx, linebuf_win, KxCHI, KxKxCHI, *pixel_iwp );
		*pixel_iwp += ( SxCHI << 1 );
	}
}


// Write back to off-chip memory
static void write_back(
	p_fmap_t *fmap_out, uint32_t *oidx, uint16_t ch_out, fmap_t *out_pix,
	uint8_t wo, uint8_t use_relu )
{
	if (wo)
	{
		L_WB_N: for ( uint16_t n = 0; n != ch_out; n+=FM_PACK )
		{
			DO_PRAGMA ( HLS loop_tripcount min=CHO_MIN_FMP max=CHO_MAX_FMP )
			DO_PRAGMA ( HLS pipeline II=1 )

			L_WB_NN: for ( uint8_t nn = 0; nn != FM_PACK; nn++ )
			{
				fmap_t tmp = ( use_relu && (out_pix[ n + nn ] < 0) ) ? 0 : out_pix[ n + nn ];
				fmap_out[ *oidx ].data[ nn ] = tmp;
			}
			(*oidx)++;
		}
	}
}



#if defined (__SIM__)

void hw_conv_fips(
		p_fmap_t fmap_in[ FMAPI_FIFO_DEPTH ],
		p_param_t param[ PARAM_FIFO_DEPTH ],
		p_fmap_t fmap_out[ FMAPO_FIFO_DEPTH ],
		uint8_t h_in, uint8_t w_in, uint16_t ch_in,
		uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
		int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu )
{
	// Interface synthsesis in case of using Vivado HLS instead of SDSoC
	/*
	DO_PRAGMA ( HLS interface ap_fifo port=fmap_in name=fmap_in )
	DO_PRAGMA ( HLS interface ap_fifo port=param name=param )
	DO_PRAGMA ( HLS interface ap_fifo port=fmap_out name=fmap_out )
	*/

#else

void hw_conv_fips(
		p_fmap_t *fmap_in,
		p_param_t *param,
		p_fmap_t *fmap_out,
		uint8_t h_in, uint8_t w_in, uint16_t ch_in,
		uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
		int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu )
{

#endif
	DO_PRAGMA ( HLS data_pack variable=fmap_in struct_level )
	DO_PRAGMA ( HLS data_pack variable=param struct_level )
	DO_PRAGMA ( HLS data_pack variable=fmap_out struct_level )


	// pre-calculate terms used frequently
	uint8_t h_out, w_out;
	uint16_t WIxCHI, KxCHI, KxKxCHI, Q_CHO, SxCHI, PADxCHI;
	uint32_t CHOxKxKxCHI, Q_CHOxKxKxCHI;
	precalc_terms(
		h_in, w_in, ch_in, ch_out, pad, kernel, stride, &h_out, &w_out,
		&WIxCHI, &KxCHI, &KxKxCHI, &Q_CHO, &SxCHI, &PADxCHI,
		&CHOxKxKxCHI, &Q_CHOxKxKxCHI );


	// Local caches

	// Parameters
	static param_t _weights[PAR_FACT][Q_CHOxKxKxCHI_MAX];
	DO_PRAGMA ( HLS array_partition variable=_weights complete dim=1 )
	DO_PRAGMA ( HLS array_partition variable=_weights cyclic factor=H_CHI_NUM dim=2 )
	DO_PRAGMA ( HLS RESOURCE variable=_weights core=RAM_2P_BRAM )

	static param_t _bias[PAR_FACT][Q_CHO_MAX];
	DO_PRAGMA ( HLS array_partition variable=_bias complete dim=1 )
	DO_PRAGMA ( HLS RESOURCE variable=_bias core=RAM_1P_LUTRAM )

	// Line buffers
	// Define input line buffers (K_MAX number of line buffers).
	// We consider the 2 last dimensions [WI_MAX][CHI_MAX]
	// as a product to save on BRAM resources.
	static fmap_t linebuf[K_MAX][WIxCHI_MAX];
	//DO_PRAGMA ( HLS array_partition variable=linebuf complete dim=1 )
	DO_PRAGMA ( HLS array_partition variable=linebuf cyclic factor=CHI_NUM dim=2 )
	DO_PRAGMA ( HLS RESOURCE variable=linebuf core=RAM_2P_BRAM )

	// We use this array as a pointer to one of the 3 line buffers
	// of the linebuf array
	static uint8_t linebuf_idx[K_MAX];
	//DO_PRAGMA ( HLS array_partition variable=linebuf_idx complete dim=1 )
	DO_PRAGMA ( HLS RESOURCE variable=linebuf_idx core=RAM_1P_LUTRAM )

	// Image windows:
	// -We use image window to be able to load a new input fmap 3D
	// pixel into linebuf while we calculate the current output fmap
	// pixel.
	// -We use 2 image windows to take advantage of double buffering.
	// -We cannot declare them in the form fmap_t linebuf_win[2][KxKxCHI_MAX]
	// and use array partitioning to divide this array into "two" arrays
	// because we need to read from it from one function and at the same
	// time write to it using another function; this functionality
	// is not currently supported by Vivado HLS.
	static fmap_t linebuf_win0[KxKxCHI_MAX];
	DO_PRAGMA ( HLS array_partition variable=linebuf_win0 cyclic factor=CHI_NUM dim=1 )
	DO_PRAGMA ( HLS RESOURCE variable=linebuf_win0 core=RAM_2P_LUTRAM )
	static fmap_t linebuf_win1[KxKxCHI_MAX];
	DO_PRAGMA ( HLS array_partition variable=linebuf_win1 cyclic factor=CHI_NUM dim=1 )
	DO_PRAGMA ( HLS RESOURCE variable=linebuf_win1 core=RAM_2P_LUTRAM )

	// Output 3D pixel buffers:
	// -We use 2 of them as part of the double buffering implementation.
	static fmap_t out_pix0[CHO_MAX];
	DO_PRAGMA ( HLS array_partition variable=out_pix0 cyclic factor=H_FM_PACK dim=1 )
	DO_PRAGMA ( HLS RESOURCE variable=out_pix0 core=RAM_2P_LUTRAM )
	static fmap_t out_pix1[CHO_MAX];
	DO_PRAGMA ( HLS array_partition variable=out_pix1 cyclic factor=H_FM_PACK dim=1 )
	DO_PRAGMA ( HLS RESOURCE variable=out_pix1 core=RAM_2P_LUTRAM )


	// Indices for input/output fmap access

	uint32_t iidx = 0;
	uint32_t oidx = 0;


	// Initialize linebuffer and parameter caches

	init_caches(
		fmap_in, &iidx, linebuf, linebuf_idx, WIxCHI, kernel, param,
		Q_CHOxKxKxCHI, Q_CHO, _weights, _bias, PADxCHI );


	// For each output row
	L_H_OUT: for ( uint8_t ho = 0; ho != h_out; ho++ )
	{
		DO_PRAGMA ( HLS loop_tripcount min=H_OUT_MIN max=H_OUT_MAX )

		// Shift linebuffer "down" in the input-feature-map.
		// Fill the "last" linebuffer line with KxCHI values (K 3D pixels).
		shift_linebuf(
			fmap_in, &iidx, linebuf, linebuf_idx, WIxCHI, KxCHI, kernel,
			stride, ho, h_out, PADxCHI );
		uint16_t lb_pixel_pt = KxCHI; // last linebuffer pixel "pointer"

		// Linebuffer window initialization
		init_linebuf_win( linebuf, linebuf_idx, linebuf_win0, KxCHI, KxKxCHI, 0 );
		// linebuffer pixel "pointers" for linebuffer windows
		uint16_t pixel_iwp0 = ( SxCHI << 1 );
		uint16_t pixel_iwp1 = SxCHI;

		// For each output 3D pixel (in each output row)
		L_W_OUT: for ( uint8_t wo = 0; wo != w_out; wo++ )
		{
			DO_PRAGMA ( HLS loop_tripcount min=W_OUT_MIN max=W_OUT_MAX )

			if ( wo%2 == 0 )
			{
				// Calc pixel
				pixel_calc( linebuf_win0, _weights, _bias, KxKxCHI,
					Q_CHOxKxKxCHI, Q_CHO, out_pix0, ei, eo, ep );

				// Update linebuffer line and linebuffer window
				update_linebuf_win( fmap_in, &iidx, linebuf, kernel,
					SxCHI, WIxCHI, &lb_pixel_pt, linebuf_idx, linebuf_win1,
					KxCHI, KxKxCHI, &pixel_iwp1, PADxCHI, ho, h_out, stride );

				// Write back to off-chip memory
				write_back( fmap_out, &oidx, ch_out, out_pix1, wo, use_relu );
			}
			else
			{
				// Calc pixel
				pixel_calc( linebuf_win1, _weights, _bias, KxKxCHI,
					Q_CHOxKxKxCHI, Q_CHO, out_pix1, ei, eo, ep );

				// Update linebuffer line and linebuffer window
				update_linebuf_win( fmap_in, &iidx, linebuf, kernel,
					SxCHI, WIxCHI, &lb_pixel_pt, linebuf_idx, linebuf_win0,
					KxCHI, KxKxCHI, &pixel_iwp0, PADxCHI, ho, h_out, stride );

				// Write back to off-chip memory
				write_back( fmap_out, &oidx, ch_out, out_pix0, wo, use_relu );
			}
		}
		// Write back to off-chip memory leftover pixel
		if ( w_out%2 == 0 )
			write_back( fmap_out, &oidx, ch_out, out_pix1, 1, use_relu );
		else
			write_back( fmap_out, &oidx, ch_out, out_pix0, 1, use_relu );
	}
}
