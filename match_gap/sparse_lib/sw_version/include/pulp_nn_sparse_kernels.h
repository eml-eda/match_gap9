#pragma once

typedef struct SPARSE_VENOM {
  int8_t * values;
  uint8_t * idxs;
  uint8_t V;
} sparse_venom;

typedef struct SPARSE8 {
  int8_t * values;
  uint8_t * idxs;
} sparse_group;

typedef struct SPARSE_CSR {
  int8_t *values;
  uint8_t *offsets;
  uint8_t *rowPointer;
} sparse_csr;

uint8_t __attribute__ ((noinline)) *pulp_nn_sparse_matmul1x2_1_1_4(
  const int8_t *pWeightVal,
  const uint8_t *pWeightIdx,
  uint8_t *      pInBufferDense,
  uint8_t *      pInBuffer,
  uint16_t       ch_out,
  uint16_t       num_col_im2col,
  uint16_t       out_shift,
  uint16_t       out_mult,
  int32_t *      k,
  int32_t *      lambda,
  const int8_t * bias,
  uint8_t *      pOut,
  uint8_t *      pOut2,
  int            flag_relu,
  int            flag_batch_norm
);



void __attribute__ ((noinline)) pulp_nn_sparse_conv_HoWo_parallel1x2_1_1_4(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  uint8_t *       pIm2ColBufferSparse,
  const int8_t    *bias,
  uint8_t *       pOutBuffer,
  const int8_t   *pWeightVal,
  const uint8_t   *pWeightIdx,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm);


void __attribute__ ((noinline)) pulp_nn_sparse_pointwise_HoWo_parallel1x2_1_1_4(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  uint8_t *       pIm2ColBufferSparse,
  const int8_t    *bias,
  uint8_t *       pOutBuffer,
  const int8_t   *pWeightVal,
  const uint8_t   *pWeightIdx,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm);


uint8_t __attribute__ ((noinline)) *pulp_nn_sparse_matmul1x2_1_1_8(
  const int8_t *pWeightVal,
  const uint8_t *pWeightIdx,
  uint8_t *      pInBufferDense,
  uint8_t *      pInBuffer,
  uint16_t       ch_out,
  uint16_t       num_col_im2col,
  uint16_t       out_shift,
  uint16_t       out_mult,
  int32_t *      k,
  int32_t *      lambda,
  const int8_t * bias,
  uint8_t *      pOut,
  uint8_t *      pOut2,
  int            flag_relu,
  int            flag_batch_norm
);



void __attribute__ ((noinline)) pulp_nn_sparse_conv_HoWo_parallel1x2_1_1_8(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  uint8_t *       pIm2ColBufferSparse,
  const int8_t    *bias,
  uint8_t *       pOutBuffer,
  const int8_t   *pWeightVal,
  const uint8_t   *pWeightIdx,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm);


uint8_t __attribute__ ((noinline)) *pulp_nn_sparse_matmul1x2_1_1_16(
  const int8_t *pWeightVal,
  const uint8_t *pWeightIdx,
  uint8_t *      pInBufferDense,
  uint8_t *      pInBuffer,
  uint16_t       ch_out,
  uint16_t       num_col_im2col,
  uint16_t       out_shift,
  uint16_t       out_mult,
  int32_t *      k,
  int32_t *      lambda,
  const int8_t * bias,
  uint8_t *      pOut,
  uint8_t *      pOut2,
  int            flag_relu,
  int            flag_batch_norm
);



void __attribute__ ((noinline)) pulp_nn_sparse_conv_HoWo_parallel1x2_1_1_16(
  const uint8_t * pInBuffer,
  uint8_t *       pIm2ColBuffer,
  uint8_t *       pIm2ColBufferSparse,
  const int8_t    *bias,
  uint8_t *       pOutBuffer,
  const int8_t   *pWeightVal,
  const uint8_t   *pWeightIdx,
  int32_t *       k,
  int32_t *       lambda,
  const uint16_t  out_mult,
  const uint16_t  out_shift,
  const uint16_t  dim_in_x,
  const uint16_t  dim_in_y,
  const uint16_t  ch_in,
  const uint16_t  dim_out_x,
  const uint16_t  dim_out_y,
  const uint16_t  ch_out,
  const uint16_t  dim_kernel_x,
  const uint16_t  dim_kernel_y,
  const uint16_t  padding_y_top,
  const uint16_t  padding_y_bottom,
  const uint16_t  padding_x_left,
  const uint16_t  padding_x_right,
  const uint16_t  stride_x,
  const uint16_t  stride_y,
  int             flag_relu,
  int             flag_batch_norm);



