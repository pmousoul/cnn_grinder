//------------------------------------------------------------------------------
//  SqueezeNetOnFPGA
//------------------------------------------------------------------------------
//
//	File:  gpool_cache.hpp
//
//  Global Pooling Cache Module for FPGA
//
//	(c) David Gschwend, 2016
//
//------------------------------------------------------------------------------

#ifndef GPool_CACHE_HPP_07571FC2
#define GPool_CACHE_HPP_07571FC2

// Data Types for FPGA Implementation
#include "fpga_top.hpp"

// ========================
// = Global Pooling Cache =
// ========================
namespace GPoolCache {

// =============
// = Variables =
// =============
data_t GBRAM[MAX_NUM_CHOUT];

// ================
// = Output Cache =
// ================

void accumulateChannel(channel_t co, data_t value_to_add) {
#pragma HLS inline

#pragma HLS FUNCTION_INSTANTIATE variable = co
#pragma HLS ARRAY_PARTITION variable = GBRAM cyclic factor = N_PE
#pragma HLS RESOURCE variable=GBRAM core=RAM_T2P_BRAM latency=2

  assert(co < MAX_NUM_CHOUT && "tried to access invalid channel");

#pragma HLS DEPENDENCE variable=GBRAM inter false
  data_t old_ch = GBRAM[co];
  data_t new_ch = old_ch + value_to_add;
  GBRAM[co] = new_ch;
  LOG("GPoolCache: accumulateChannel( ch%-2d ) add %+.2f -> %.2f\n", (int)co,
      value_to_add, new_ch);
}

data_t getChannel(channel_t c) {
#pragma HLS inline
  assert(c < MAX_NUM_CHOUT && "tried to read invalid channel");
  if (LOG_DETAILS)
    LOG("GPoolCache: getChannel( ch%-2d ) -> %6.2f\n", (int)c, GBRAM[c]);
  return GBRAM[c];
}

void setChannel(channel_t c, data_t data) {
#pragma HLS inline
  assert(c < MAX_NUM_CHOUT && "tried to write invalid channel");
  if (LOG_DETAILS)
    LOG("GPoolCache: setChannel( ch%-2d ) <- %6.2f\n", (int)c, data);
  GBRAM[c] = data;
}

};

#endif /* end of include guard: OUTPUT_CACHE_HPP_07571FC2 */
