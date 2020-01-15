// Include file for HW function implementation

#ifndef HW_FUNC_DFP_HPP
#define HW_FUNC_DFP_HPP


// include files

// The line bellow is used for Vivado HLS c/rtl co-simulation
// Change this according to your system
// It might be unnecessary for earlier Vivado HLS versions
#if defined(__SIM__)
#include "/home/pmousoul/Desktop/pmousoul_data/sw/Xilinx_SDx/Vivado/2018.2/include/gmp.h"
#endif

#include <cstdint>   // for constant length (u)int types
#include "debug.hpp" // for debugging

#include "layer_args.hpp"
#include "fp_mul_pow2.hpp"
#include <math.h>


// Preprocessor code to use parameters in the HLS pragmas
#define DO_PRAGMA_INNER(x) _Pragma (#x)
#define DO_PRAGMA(x) DO_PRAGMA_INNER(x)


// Constant parameters

// We use packed feature map and parameter data
#define FM_PACK   8
#define H_FM_PACK 4
#define WB_PACK   8

// FIFO depth for ap_fifo interfaces used in Vivado HLS
// These are the max fifo depths considering that 64-bit
// (8 8-bit) values are transfered
#define FMAPI_FIFO_DEPTH 131072
#define PARAM_FIFO_DEPTH  32288
#define FMAPO_FIFO_DEPTH 131072

// Number of input channels
#define CHI_NUM   16
#define H_CHI_NUM  8

// Parallel factor - number of PEs
#define PAR_FACT  16
#define DSP_FACT   8 // Number of PEs implemented with DSPs
#define BIT_SHIFT  4 // Output channels are computed by PAR_FACT PEs

// Kernel sizes
#define K_MIN 1
#define K_MAX 3

// Feature map sizes
#define H_OUT_MIN   8
#define H_OUT_MAX 128
#define W_OUT_MIN   8
#define W_OUT_MAX 128

// Parameter related constants
// WBP at the end means that it is divided by WB_PACK
// CHN at the end means that it is divided by CHI_NUM
// Q at the beggining means that is is divided by PAR_FACT
#define CHOxKxKxCHI_MAX       258048
#define CHOxKxKxCHI_MAX_WBP    32256
#define CHOxKxKxCHI_MIN         1024
#define Q_CHOxKxKxCHI_MAX      16128
#define Q_CHOxKxKxCHI_MAX_WBP   2016
#define Q_CHOxKxKxCHI_MAX_CHN   1008
#define Q_CHOxKxKxCHI_MIN         64
#define Q_CHOxKxKxCHI_MIN_WBP      8
#define Q_CHOxKxKxCHI_MIN_CHN      4

// Output channel related constants
// FMP at the end means that it is divided by FM_PACK
#define CHO_MAX     368
#define CHO_MAX_WBP  46
#define CHO_MAX_FMP  46
#define Q_CHO_MAX    23
#define CHO_MIN      16
#define CHO_MIN_WBP   2
#define CHO_MIN_FMP   2
#define Q_CHO_MIN     1

// Input channel related constants
#define CHI_MAX     736
#define CHI_MIN      16
#define CHI_MAX_FMP  92
#define CHI_MIN_FMP   2

// Activation line related constants
#define WIxCHI_MAX     8704
#define WIxCHI_MAX_FMP 1088
#define WIxCHI_MIN      896
#define WIxCHI_MIN_FMP  112

// kernel window related constants 
#define KxCHI_MAX     1152
#define KxCHI_MAX_FMP  144
#define KxCHI_MIN       16
#define KxCHI_MIN_FMP    2

#define KxKxCHI_MAX     3456
#define KxKxCHI_MAX_CHN  216
#define KxKxCHI_MIN       16
#define KxKxCHI_MIN_CHN    1


// Mpool related constants
#define PFMAP_FIFO_DEPTH 65536

// Activations line related constant
#define PWIxPCH_MAX     7232
#define PWIxPCH_MIN     3584
#define PWIxPCH_MAX_FMP  904
#define PWIxPCH_MIN_FMP  448

#define PCH_MAX    128
#define PCH_MIN     64
#define PCH_MAX_FMP 16
#define PCH_MIN_FMP  8

#define PHOxPWOxPCH_MIN        3072
#define PHOxPWOxPCH_MAX     1048576
#define PHOxPWOxPCH_MIN_FMP     384
#define PHOxPWOxPCH_MAX_FMP  131072

#define PHO_MIN 14
#define PHO_MAX 56
#define PWO_MIN 14
#define PWO_MAX 56


// Fixed-point type definitions and packed i/o conv data 

typedef int8_t fmap_t;  // feature map
typedef int8_t param_t; // weights, bias

// multiplication of fmap_t and param_t
typedef int16_t mul_t;

// accumulation of multiple mul_t
typedef int32_t acc_t;

typedef struct p_fmap_t_struct{
	int8_t data[FM_PACK];
} __attribute__ ((packed, aligned(4))) p_fmap_t;
typedef struct p_param_t_struct{
	int8_t data[WB_PACK];
} __attribute__ ((packed, aligned(4))) p_param_t;

#if defined (__SDSCC__)

#if defined (USE_AFI)

	#if defined(ULTRA96)
		#pragma SDS data sys_port( fmap_in:  AFI )
	#else
		#pragma SDS data sys_port( fmap_in:  ACP )
	#endif

	#pragma SDS data sys_port( param:    AFI )
	
	#if defined(ULTRA96)
		#pragma SDS data sys_port( pfmap_out:  AFI )
	#else
		#pragma SDS data sys_port( pfmap_out:  ACP )
	#endif
	
	#pragma SDS data copy( fmap_in[0: ( h_in * w_in * ch_in ) / FM_PACK ] )
	#pragma SDS data copy( param[0: ( ch_out * kernel * kernel * ch_in + ch_out ) / WB_PACK ] )
	#pragma SDS data copy( pfmap_out[0: ( ( ( ( ( ( (h_in - kernel + (pad<<1))>>(stride-1) ) + 1 ) - pkernel + ppad)>>(pstride-1) ) + 1 ) * ( ( ( ( ( (w_in - kernel + (pad<<1))>>(stride-1) ) + 1 ) - pkernel + ppad)>>(pstride-1) ) + 1 ) * ch_out ) / FM_PACK ] )
	#pragma SDS data access_pattern( fmap_in:  SEQUENTIAL )
	#pragma SDS data access_pattern( param:    SEQUENTIAL )
	#pragma SDS data access_pattern( pfmap_out: SEQUENTIAL )
	#pragma SDS data data_mover( fmap_in:  AXIDMA_SIMPLE )
	#pragma SDS data data_mover( param:    AXIDMA_SIMPLE )
	#pragma SDS data data_mover( pfmap_out: AXIDMA_SIMPLE )
	#pragma SDS data mem_attribute( fmap_in:  PHYSICAL_CONTIGUOUS )
	
	#if defined(ULTRA96)
		#pragma SDS data mem_attribute( param:    PHYSICAL_CONTIGUOUS )
	#else
		#pragma SDS data mem_attribute( param:    PHYSICAL_CONTIGUOUS|NON_CACHEABLE )
	#endif
	
	#pragma SDS data mem_attribute( pfmap_out: PHYSICAL_CONTIGUOUS )

	void hw_conv_mpool_dfp( p_fmap_t *fmap_in, p_param_t *param,
		p_fmap_t *pfmap_out, uint8_t h_in, uint8_t w_in, uint16_t ch_in,
		uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
		int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu,
		uint8_t ppad, uint8_t pkernel, uint8_t pstride );

#else

	#pragma SDS data sys_port( fmap_in:  ACP )
	#pragma SDS data sys_port( param:    ACP )
	#pragma SDS data sys_port( pfmap_out: ACP )
	#pragma SDS data copy( fmap_in[0: ( h_in * w_in * ch_in ) / FM_PACK ] )
	#pragma SDS data copy( param[0: ( ch_out * kernel * kernel * ch_in + ch_out ) / WB_PACK ] )
	#pragma SDS data copy( pfmap_out[0: ( ( ( ( ( ( (h_in - kernel + (pad<<1))>>(stride-1) ) + 1 ) - pkernel + ppad)>>(pstride-1) ) + 1 ) * ( ( ( ( ( (w_in - kernel + (pad<<1))>>(stride-1) ) + 1 ) - pkernel + ppad)>>(pstride-1) ) + 1 ) * ch_out ) / FM_PACK ] )
	#pragma SDS data access_pattern( fmap_in:  SEQUENTIAL )
	#pragma SDS data access_pattern( param:    SEQUENTIAL )
	#pragma SDS data access_pattern( pfmap_out: SEQUENTIAL )
	#pragma SDS data data_mover( fmap_in:  AXIDMA_SIMPLE )
	#pragma SDS data data_mover( param:    AXIDMA_SIMPLE )
	#pragma SDS data data_mover( pfmap_out: AXIDMA_SIMPLE )
	#pragma SDS data mem_attribute( fmap_in:  PHYSICAL_CONTIGUOUS )
	#pragma SDS data mem_attribute( param:    PHYSICAL_CONTIGUOUS )
	#pragma SDS data mem_attribute( pfmap_out: PHYSICAL_CONTIGUOUS )

	void hw_conv_mpool_dfp( p_fmap_t *fmap_in, p_param_t *param,
		p_fmap_t *pfmap_out, uint8_t h_in, uint8_t w_in, uint16_t ch_in,
		uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
		int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu,
		uint8_t ppad, uint8_t pkernel, uint8_t pstride );

#endif

#elif defined (__SIM__)

	void hw_conv_mpool_dfp(
		p_fmap_t fmap_in[ FMAPI_FIFO_DEPTH ],
		p_param_t param[ PARAM_FIFO_DEPTH ],
		p_fmap_t pfmap_out[ FMAPO_FIFO_DEPTH ],
		uint8_t h_in, uint8_t w_in, uint16_t ch_in,
		uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
		int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu,
		uint8_t ppad, uint8_t pkernel, uint8_t pstride );

#else

	void hw_conv_mpool_dfp(
		p_fmap_t *fmap_in,
		p_param_t *param,
		p_fmap_t *pfmap_out,
		uint8_t h_in, uint8_t w_in, uint16_t ch_in,
		uint16_t ch_out, uint8_t pad, uint8_t kernel, uint8_t stride,
		int8_t ei, int8_t eo, int8_t ep, uint8_t use_relu,
		uint8_t ppad, uint8_t pkernel, uint8_t pstride );

#endif


// SqN v1.1 HW Fixed-point implementation
void hw_sqn_dfp(
	p_fmap_t* hw_fmap_dfp[4], p_param_t* hw_params_dfp[SQN_LN], float* hw_fmap_flp_o, Largs_sqn& larg, float error_sz, bool* error_flag );

// ZynqNet HW Fixed-point implementation
void hw_zqn_dfp(
	p_fmap_t* hw_fmap_dfp[4], p_param_t* hw_params_dfp[ZQN_LN], float* hw_fmap_flp_o, Largs_zqn& larg, float error_sz, bool* error_flag );

#endif
