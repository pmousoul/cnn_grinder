// HW implementation of the maxpool layer

#include "hw_func_fips.hpp"

// Fixed-point implementation of the max-pool layer
// Zero-pading the input is done on the fly
#if defined (__SIM__)

void hw_maxpool_fips(
	p_fmap_t pfmap_in[ FMAPI_FIFO_DEPTH ],
	p_fmap_t pfmap_out[ FMAPO_FIFO_DEPTH ],
	uint8_t ph_in, uint8_t pw_in, uint16_t pch_in,
	uint8_t ppad, uint8_t pkernel, uint8_t pstride )
{

#else

void hw_maxpool_fips(
	p_fmap_t *pfmap_in, p_fmap_t *pfmap_out,
	uint8_t ph_in, uint8_t pw_in, uint16_t pch_in,
	uint8_t ppad, uint8_t pkernel, uint8_t pstride )
{

#endif
	DO_PRAGMA ( HLS data_pack variable=pfmap_in struct_level )
	DO_PRAGMA ( HLS data_pack variable=pfmap_out struct_level )

	// Calculate output fmap dimensions
	uint8_t PWO=( (pw_in-pkernel+ppad)>>(pstride-1) ) + 1;
	uint8_t PHO=( (ph_in-pkernel+ppad)>>(pstride-1) ) + 1;

		// Pre-calculate some frequently used variables
	uint16_t PCH     = pch_in;
	uint16_t PWOxPCH = PWO   * PCH;
	uint16_t PWIxPCH = pw_in * PCH;

	// Local caches

	// Current row
	fmap_t cur_row[ PWIxPCH_MAX ];
	DO_PRAGMA ( HLS array_partition variable=cur_row cyclic factor=FM_PACK dim=1 )
	// Result row
	fmap_t res_row[ PWIxPCH_MAX ];
	DO_PRAGMA ( HLS array_partition variable=res_row cyclic factor=FM_PACK dim=1 )
	// Current pixel
	fmap_t cur_pixel[ PCH_MAX ];
	DO_PRAGMA ( HLS array_partition variable=cur_pixel cyclic factor=FM_PACK dim=1 )
	// Result pixel
	fmap_t res_pixel[ PCH_MAX ];
	DO_PRAGMA ( HLS array_partition variable=res_pixel cyclic factor=FM_PACK dim=1 )

	// Indices for input/output fmap access
	uint32_t iidx = 0;
	uint32_t oidx = 0;

	// Initialize current row and perform max-pooling
	if ( pkernel != 1 )
	{
		// Perform max-pooling
		L_MPOOL: for (uint8_t pho=0; pho<PHO; pho++)
		{
			DO_PRAGMA ( HLS loop_tripcount min=PHO_MIN max=PHO_MAX )

			uint8_t wh_s = pho*pstride;
			uint8_t wh_e = pho*pstride + pkernel;

			// perform maxpool between rows
			L_ROW_H: for (uint8_t wh = wh_s; wh<wh_e; wh++)
			{
				DO_PRAGMA ( HLS loop_tripcount min=K_MAX max=K_MAX )

				L_ROW_CH: for (uint16_t pwixpch=0; pwixpch<PWIxPCH; pwixpch+=FM_PACK)
				{
					DO_PRAGMA ( HLS loop_tripcount min=PWIxPCH_MIN_FMP  max=PWIxPCH_MAX_FMP  )
					DO_PRAGMA ( HLS pipeline II=1 )

					uint16_t n_tmp = pwixpch / FM_PACK;

					// we might have to zero-pad the input at the bottom
					if (wh < ph_in)
					{
						if ( ( wh != wh_s ) || ( pho == 0 ) )
						{
							L_READ_INPUT: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
								cur_row[ n_tmp*FM_PACK + nn ] = pfmap_in[ iidx ].data[nn];
							iidx++;
						}
					}
					else
					{
						L_READ_ZERO_R: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
							cur_row[ n_tmp*FM_PACK + nn ] = 0;
					}

					L_ROW_CHI_N: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
					{
						if (wh == wh_s)
							res_row[ n_tmp*FM_PACK + nn ] = cur_row[ n_tmp*FM_PACK + nn ];
						else
							res_row[ n_tmp*FM_PACK + nn ] =
								( res_row[ n_tmp*FM_PACK + nn ] < cur_row[ n_tmp*FM_PACK + nn ] ) ?
									cur_row[ n_tmp*FM_PACK + nn ] : res_row[ n_tmp*FM_PACK + nn ];
					}

				}
			}

			// perform maxpool between columns
			L_COL_W: for (uint8_t pwo=0; pwo<PWO; pwo++)
			{
				DO_PRAGMA ( HLS loop_tripcount min=PWO_MIN max=PWO_MAX )

				uint8_t ww_s = pwo*pstride;
				uint8_t ww_e = pwo*pstride + pkernel;

				L_COL_WW: for (uint8_t ww = ww_s; ww<ww_e; ww++)
				{
					DO_PRAGMA ( HLS loop_tripcount min=K_MAX max=K_MAX )

					L_COL_WW_CH: for (uint16_t pch=0; pch<PCH; pch+=FM_PACK)
					{
						DO_PRAGMA ( HLS loop_tripcount min=PCH_MIN_FMP max=PCH_MAX_FMP )
						DO_PRAGMA ( HLS pipeline II=1 )

						uint16_t n_tmp = pch / FM_PACK;
						uint16_t n_tmp2 = ( ww*PCH ) / FM_PACK;

						// we might have to zero-pad at the right
						if (ww < pw_in)
						{
							if ( ( ww != ww_s ) || ( pwo == 0 ) )
								L_READ_CUR_PIX: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
									cur_pixel[ n_tmp*FM_PACK + nn ] = res_row[ n_tmp2*FM_PACK + n_tmp*FM_PACK + nn ];
						}
						else
						{
							L_READ_ZERO_C: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
								cur_pixel[ n_tmp*FM_PACK + nn ] = 0;
						}

						L_COL_WW_CHI_N: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
						{
							if (ww == ww_s)
								res_pixel[ n_tmp*FM_PACK + nn ] = cur_pixel[ n_tmp*FM_PACK + nn ];
							else
								res_pixel[ n_tmp*FM_PACK + nn ] =
									( res_pixel[ n_tmp*FM_PACK + nn ] < cur_pixel[ n_tmp*FM_PACK + nn ] ) ?
										cur_pixel[ n_tmp*FM_PACK + nn ] : res_pixel[ n_tmp*FM_PACK + nn ];

						}

					}

				}

				// write back pixel
				L_CH_OUT: for (uint16_t pch=0; pch<PCH; pch+=FM_PACK)
				{
					DO_PRAGMA ( HLS loop_tripcount min=PCH_MIN_FMP max=PCH_MAX_FMP )
					DO_PRAGMA ( HLS pipeline II=1 )

					uint16_t n_tmp = pch / FM_PACK;

					L_CH_OUT_N: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
					{
						pfmap_out[ oidx ].data[ nn ] = res_pixel[ n_tmp*FM_PACK + nn ];
					}

					(oidx)++;
				}

			}
		}
	}
	else
	{
		// Bypass the maxpool unit
		uint32_t PHOxPWOxPCH = PHO*PWO*PCH;
		L_BYPASS: for (uint32_t idx=0; idx<PHOxPWOxPCH/FM_PACK; idx++ )
		{
			DO_PRAGMA ( HLS loop_tripcount min=PHOxPWOxPCH_MIN_FMP max=PHOxPWOxPCH_MAX_FMP )
			DO_PRAGMA ( HLS pipeline II=1 )

			L_BYPASS_NN: for ( uint8_t nn = 0; nn != FM_PACK; nn++)
			{
				pfmap_out[ idx ].data[ nn ] = pfmap_in[ idx ].data[ nn ];
			}
		}
	}

}
