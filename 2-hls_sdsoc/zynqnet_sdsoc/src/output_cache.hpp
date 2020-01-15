//------------------------------------------------------------------------------
//  SqueezeNetOnFPGA
//------------------------------------------------------------------------------
//
//	File:  output_cache.hpp
//
//  Output Cache Module for FPGA
//
//	(c) David Gschwend, 2016
//
//------------------------------------------------------------------------------

#ifndef OUTPUT_CACHE_HPP_07571FC2
#define OUTPUT_CACHE_HPP_07571FC2

// Data Types for FPGA Implementation
#include "fpga_top.hpp"

// ================
// = Output Cache =
// ================
namespace OutputCache {

  // =============
// = Variables =
// =============

data_t OBRAM[MAX_NUM_CHOUT];

// ================
// = Output Cache =
// ================

data_t getChannel(channel_t co) {
#pragma HLS inline
  assert(co < MAX_NUM_CHOUT && "tried to read invalid channel");
  if (LOG_DETAILS)
    LOG("OCache: getChannel( ch%-2d ) -> %6.2f\n", (int)co, OBRAM[co]);
  return OBRAM[co];
}

void setChannel(channel_t co, data_t data) {
#pragma HLS inline
#pragma HLS FUNCTION_INSTANTIATE variable = co

  assert(co < MAX_NUM_CHOUT && "tried to write invalid channel");
  if (LOG_DETAILS)
    LOG("OCache: setChannel( ch%-2d ) <- %6.2f\n", (int)co, data);
  OBRAM[co] = data;
}

void accumulateChannel(channel_t co, data_t value_to_add) {
#pragma HLS inline
//#pragma HLS pipeline

#pragma HLS FUNCTION_INSTANTIATE variable = co
#pragma HLS ARRAY_PARTITION variable = OBRAM cyclic factor = N_PE
#pragma HLS RESOURCE variable=OBRAM core=RAM_T2P_BRAM latency=2

  assert(co < MAX_NUM_CHOUT && "tried to access invalid channel");

  data_t old_ch = getChannel(co); /* BRAM[c] */
  data_t new_ch = old_ch + value_to_add;
  setChannel(co, new_ch); /* BRAM[c] = new_ch; */
  LOG("OCache: accumulateChannel( ch%-2d ) add %+.2f -> %.2f\n", (int)co,
      value_to_add, new_ch);
}

};

#endif /* end of include guard: OUTPUT_CACHE_HPP_07571FC2 */
