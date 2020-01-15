//------------------------------------------------------------------------------
//  SqueezeNetOnFPGA
//------------------------------------------------------------------------------
//
//	File:  memory_controller.hpp
//
//  Memory Controller Module for FPGA (connects to AXI_M bus --> DMA)
//
//	(c) David Gschwend, 2016
//
//------------------------------------------------------------------------------

#ifndef MEMORY_CONTROLLER_HPP_EBD6F5A3
#define MEMORY_CONTROLLER_HPP_EBD6F5A3

// Data Types for FPGA Implementation
#include "fpga_top.hpp"
#include "output_cache.hpp"
#include "gpool_cache.hpp"

// =====================
// = Memory Controller =
// =====================
namespace MemoryController {

// =============
// = Variables =
// =============

memaddr_t pixel_output_offset;
memaddr_t layer_weights_offset;
memaddr_t layer_input_offset;
memaddr_t layer_output_offset;
memaddr_t layer_pixel_offset;
pixelperrow_t pixels_per_row;
dimension_t width_out;
channel_t ch_out;
unsigned int dram_weights_offset;
unsigned int dram_data_offset;
bool is_first_split_layer;
bool is_second_split_layer;

// =====================
// = Memory Controller =
// =====================
void setup(data_t* SHARED_DRAM, unsigned int weights_offset,
                             unsigned int data_offset) {
#pragma HLS inline

  dram_weights_offset = (weights_offset);
  dram_data_offset = (data_offset);

  LOG("MemoryCtrl: Constructor.\n");
  LOG(" - SHARED_DRAM     = %lu\n", (long)SHARED_DRAM);
  LOG(" - DRAM_WEIGHTS    = @%luB\n", (long)dram_weights_offset);
  LOG(" - DRAM_DATA       = @%luB\n", (long)dram_data_offset);
}

void setLayerConfig(layer_t& layer) {
  layer_weights_offset = layer.mem_addr_weights;
  layer_input_offset = layer.mem_addr_input;
  layer_output_offset = layer.mem_addr_output;
  pixels_per_row = layer.width * layer.channels_in;
  ch_out = layer.channels_out;
  width_out = (layer.stride == 2) ? (layer.width / 2) : (layer.width / 1);
  is_first_split_layer = layer.is_first_split_layer;
  is_second_split_layer = layer.is_second_split_layer;

  LOG("MemoryCtrl: setLayerConfig.\n");
  LOG(" - weights offset  = %6d Elements, DRAM_WEIGHTS @%8luB\n",
      (int)layer_weights_offset, (int)layer_weights_offset * sizeof(data_t));
  LOG(" - input offset    = %6d Elements, DRAM_DATA    @%8luB\n",
      (int)layer_input_offset, (int)layer_input_offset * sizeof(data_t));
  LOG(" - output offset   = %6d Elements, DRAM_DATA    @%8luB\n",
      (int)layer_output_offset, (int)layer_output_offset * sizeof(data_t));
  LOG(" - pixels per row  = %d\n", (int)pixels_per_row);
  LOG(" - ch_out          = %d\n", (int)ch_out);
  LOG(" - width_out       = %d\n", (int)width_out);
  LOG(" - is_first_split_layer = %d\n", (int)is_second_split_layer);
  LOG(" - is_second_split_layer = %d\n", (int)is_first_split_layer);
}

data_t loadNextWeight(data_t* SHARED_DRAM,
                                        weightaddr_t addr) {
#pragma HLS inline
#pragma HLS pipeline
  data_t read = reg(SHARED_DRAM[dram_weights_offset + layer_weights_offset + addr]);
  if (LOG_DETAILS)
    LOG("MemoryCtrl: loadNextWeight  (#%4d from DRAM @%4luB): %6.2f\n",
        (int)layer_weights_offset, (long)addr, read);
  return read;
}

void setPixelLoadRow(coordinate_t y) {
  LOG("MemoryCtrl: setPixelLoadRow (row %2d).\n", (int)y);
  layer_pixel_offset = layer_input_offset + pixels_per_row * y;
}

data_t loadNextChannel(data_t* SHARED_DRAM) {
#pragma HLS inline
#pragma HLS pipeline II=1
  data_t pixel_from_ram = reg(SHARED_DRAM[dram_data_offset + layer_pixel_offset]);
  if (LOG_DETAILS)
    LOG("MemoryCtrl: loadNextChannel (from DRAM @%4luB) -> %.2f\n",
        (int)layer_pixel_offset * sizeof(data_t), pixel_from_ram);
  layer_pixel_offset++;  // increment address for next fetch
  return pixel_from_ram;
};

void setupPixelWriteback(coordinate_t y_out,
                                           coordinate_t x_out) {
#pragma HLS inline
#pragma HLS pipeline

  // Calculate Output Memory Address
  memaddr_t y_offset = y_out * width_out;
#pragma HLS RESOURCE variable = y_offset core = MulnS latency = 2
  memaddr_t xy_offset = y_offset + x_out;
  memaddr_t px_offset = xy_offset * ch_out;
#pragma HLS RESOURCE variable = px_offset core = MulnS latency = 2

  // Leave double space for "expand" layers (more ch_out will be added)
  bool is_split_layer = (is_first_split_layer | is_second_split_layer);
  pixel_output_offset = layer_output_offset +
                        (is_split_layer ? 2 * (int)px_offset : (int)px_offset);

  LOG("MemoryController: setupPixelWriteback (%2d, %2d)\n", (int)y_out,
      (int)x_out);

  LOG(" - writing %2d channels to DRAM @%luB+\n", (int)ch_out,
      (long)pixel_output_offset);
}

void writeBackOutputChannel(data_t* SHARED_DRAM, channel_t co,
                                              data_t data) {
#pragma HLS inline
  LOG_LEVEL_INCR;
  SHARED_DRAM[dram_data_offset + pixel_output_offset + co] = data;
  LOG(" WB ch%d [@%lu]: %6.2f\n", (int)co, (long)(pixel_output_offset + co),
      data);
  LOG_LEVEL_DECR;
}

void writeBackResult(data_t* SHARED_DRAM) {
#pragma HLS inline

  memaddr_t split_offset = (is_second_split_layer) ? (int)ch_out : 0;
  LOG("MemoryCtrl: writeBackResult (%d Bytes) to DRAM @%luB\n",
      (int)(ch_out * sizeof(data_t)), (long)split_offset);

L_writeBackResult:
  for (int i = 0; i < ch_out; i++) {  // ch_out set from last layer
#pragma HLS LOOP_TRIPCOUNT min = 1000 max = 1024 avg = 1000
#pragma HLS pipeline
    SHARED_DRAM[dram_data_offset + split_offset + i] =
        GPoolCache::getChannel(i);
    LOG(" WB ch%d (@%luB): %6.2f\n", (int)i, (long)(split_offset + i),
        GPoolCache::getChannel(i));
  }
}

};

#endif /* end of include guard: MEMORY_CONTROLLER_HPP_EBD6F5A3 */
