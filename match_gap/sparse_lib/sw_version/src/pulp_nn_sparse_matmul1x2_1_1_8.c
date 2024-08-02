
#include "pmsis.h"
#include "pulp_nn_sparse_kernels.h"
#include "pulp_nn_utils.h"

#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x) __builtin_pulp_clipu_r(x, 255)
#define M 4
#define LOGM 2
#define V 1
#define N 1
#define GET_IDX(mask, shift) ((mask >> shift) & 0xF)
#define MOVE8(x) (((x+1)>>1))



uint8_t __attribute__((noinline)) *
    pulp_nn_sparse_matmul1x2_1_1_8(
                                   const int8_t *pWeightVal,
                                   const uint8_t *pWeightIdx,
                                   uint8_t* pInBufferDense,
                                   uint8_t* pInBuffer,
                                   uint16_t ch_out,
                                   uint16_t num_col_im2col,
                                   uint16_t out_shift,
                                   uint16_t out_mult,
                                   int32_t* k,
                                   int32_t* lambda,
                                   const int8_t* bias,
                                   uint8_t* pOut1,
                                   uint8_t* pOut2,
                                   int flag_relu,
                                   int flag_batch_norm) {
  int8_t* pA1 = pWeightVal;
  uint8_t* pMask = pWeightIdx;

  v4s vecA1;
  v4u vecB1;
  v4u vecB2;

  uint8_t maschera_decimate;
  // Numero di 8bit per gli indici
  // x 4 -> numero di bit
  // +7 >> 3, arrotondo a 8
  int shift = ((num_col_im2col<<2) + 7)>>3;

  for (int i_ch_out = 0; i_ch_out < ch_out; i_ch_out += 1) {
    uint8_t *pMask = pWeightIdx + i_ch_out * shift;
    pA1 = pWeightVal + i_ch_out * num_col_im2col;
    /* this loop over the OFM channels */
    uint8_t* pB1 = pInBufferDense;
    uint8_t* pB2 = (pB1 + (num_col_im2col << 3));

    int bias1 = 0;
    int bias2 = 0;

    if (bias != NULL) {
      bias1 = ((int)(*bias++));
      bias2 = bias1;
    }

    /* init the accumulators with corresponding biases. if there is the
     * accumulation buffer, i included it here */
    int sum1 = bias1;
    int sum2 = bias2;

    uint16_t col_cnt_im2col = num_col_im2col & 0x3;

    for (int j = 0; j<num_col_im2col>>2; j++) {
      maschera_decimate = *(pMask++);
      int position1 = GET_IDX(maschera_decimate, 0);
      int position2 = GET_IDX(maschera_decimate, 4);
      maschera_decimate = *(pMask++);
      int position3 = GET_IDX(maschera_decimate, 0);
      int position4 = GET_IDX(maschera_decimate, 4);
      // Carico i pesi
      vecA1 = *((v4s*)pA1);

      vecB1 =  (v4u) { pB1[position1], pB1[8+position2], pB1[16 + position3], pB1[24 + position4]};
      vecB2 =  (v4u) { pB2[position1], pB2[8+position2], pB2[16 + position3], pB2[24 + position4]};


      // Dot product
      sum1 = SumDotp(vecB1, vecA1, sum1);
      sum2 = SumDotp(vecB2, vecA1, sum2);
      // Sposto  i pesi
      pA1 += 4;
      pB1 += 32;
      pB2 += 32;
    }

    int leftover_instruction_calls = (num_col_im2col & 0x3);
    
    if (leftover_instruction_calls >= 2) {
      maschera_decimate = *(pMask++);
      // 1 load - due blocchi, i pesi sono paddati a x8 
      int8_t inA1 = *(pA1++);
      int position = GET_IDX(maschera_decimate, 0);

      uint8_t     inB1  = pB1[position];
      uint8_t     inB2  = pB2[position];

      sum1  += inA1  * inB1;
      sum2 += inA1 * inB2;
      pB1 += 8;
      pB2 += 8;

      inA1 = *(pA1++);
      position = GET_IDX(maschera_decimate, 4);
      inB1 = pB1[position];
      inB2 = pB2[position];

      sum1  += inA1  * inB1;
      sum2 += inA1 * inB2;

      // Decimo gli input
      leftover_instruction_calls-=2;
      pB1 += 8;
      pB2 += 8;
    }
    if (leftover_instruction_calls == 1) {
      maschera_decimate = *(pMask++);
      // 1 load - due blocchi, i pesi sono paddati a x8 
      int8_t inA1 = *(pA1++);
      int position = GET_IDX(maschera_decimate, 0);

      uint8_t     inB1  = pB1[position];
      uint8_t     inB2  = pB2[position];

      sum1  += inA1  * inB1;
      sum2 += inA1 * inB2;
      pB1 += 8;
      pB2 += 8;
      leftover_instruction_calls-=1;
    }


    if (flag_batch_norm && flag_relu) {
      *pOut1 = pulp_nn_bn_quant_u8(sum1, *k, *lambda, out_shift);
      pOut1++;
      *pOut2 = pulp_nn_bn_quant_u8(sum2, *k, *lambda, out_shift);
      pOut2++;
      k++;
      lambda++;
    } else {
      if (flag_relu == 1) {
        *pOut1 = pulp_nn_quant_u8(sum1, out_mult, out_shift);
        pOut1++;
        *pOut2 = pulp_nn_quant_u8(sum2, out_mult, out_shift);
        pOut2++;

      } else {
        *pOut1 = (uint8_t)clip8(sum1 >> out_shift);
        pOut1++;
        *pOut2 = (uint8_t)clip8(sum2 >> out_shift);
        pOut2++;
      }
    }
  }
  pOut1 += ch_out;
  return pOut1;
}