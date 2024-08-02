
#include "pmsis.h"
#include "pulp_nn_sparse_kernels.h"
#include "pulp_nn_utils.h"

#define SumDotp(a, b, c) __builtin_pulp_sdotusp4(a, b, c)
#define clip8(x) __builtin_pulp_clipu_r(x, 255)
#define M 16
#define LOGM 2
#define V 1
#define N 1


#define __decimate1_4(x, y, z)                                  \
  asm("pv.decimate.b.llw %0, %1, %2" : "+r"(z) : "r"(x), "r"(y)); \
  asm volatile("" ::: "memory")

#define __decimate1_16_block8(vecB1, vecB2, addr_B1, addr_B2, mask) \
  asm("pv.decimate.b.llw %0, %2, %4\n"                               \
      "pv.decimate.b.llw %1, %3, %4\n"                               \
      "pv.decimate.b.llw %0, %2, %4\n"                               \
      "pv.decimate.b.llw %1, %3, %4\n"                               \
      "pv.decimate.b.llw %0, %2, %4\n"                               \
      "pv.decimate.b.llw %1, %3, %4\n"                               \
      "pv.decimate.b.llw %0, %2, %4\n"                               \
      "pv.decimate.b.llw %1, %3, %4\n"                               \
      : "+r"(vecB1), "+r"(vecB2)                                   \
      : "r"(addr_B1), "r"(addr_B2), "r"(mask))
#define __decimate1_16_block2(vecB1, vecB2, addr_B1, addr_B2, mask) \
  asm("pv.decimate.b.llw %0, %2, %4\n"                               \
      "pv.decimate.b.llw %1, %3, %4\n"                               \
      : "+r"(vecB1), "+r"(vecB2)                                   \
      : "r"(addr_B1), "r"(addr_B2), "r"(mask))

#define __reset_csr(x)               \
  { asm("fssr %0\n" : : "r"((x))); } \
  while (0)


uint8_t __attribute__((noinline)) *
    pulp_nn_sparse_matmul1x2_1_1_16(
                                   const int8_t *pWeightVal,
                                   const uint32_t *pWeightIdx,
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
  #if defined(NOISA)
  int csr = 0;
  #endif
  int8_t* pA1 = pWeightVal;
  uint32_t* pMask = pWeightIdx;

  v4s vecA1;
  v4u vecB1;
  v4u vecB2;

  uint32_t maschera_decimate;
  // Numero di 8bit per gli indici
  // numcolim2col * 2 - > indici replicati
  // x 4 -> numero di bit
  // +7 >> 3, arrotondo a 8
  int shift = ((num_col_im2col<<4) + 63)>>(4 + 2);
  //int shift = ((num_col_im2col<<4) + 15)>>(4 + 2);

  for (int i_ch_out = 0; i_ch_out < ch_out; i_ch_out += 1) {
    __reset_csr(0);
    uint32_t *pMask = pWeightIdx + i_ch_out * shift;
    /* this loop over the OFM channels */
    uint8_t* pB1 = pInBufferDense;
    uint8_t* pB2 = (pB1 + (num_col_im2col << 4));

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
      maschera_decimate = *(pMask);
      // Carico i pesi
      vecA1 = *((v4s*)pA1);
      // Decimo gli input
      __decimate1_16_block8(vecB1, vecB2, pB1, pB2, maschera_decimate);
      // Dot product
      sum1 = SumDotp(vecB1, vecA1, sum1);
      sum2 = SumDotp(vecB2, vecA1, sum2);
      // Sposto  i pesi
      pA1 += 4;
      pMask += 1;
    }

    int leftover_instruction_calls = (num_col_im2col & 0x3);
    if (leftover_instruction_calls) {
      maschera_decimate = *(pMask);
    }
    
    int contatore_shift = 0;  // Decimate scrive in posizioni diverse di vecB,
                              // prendo l'elemento giusto
    while (leftover_instruction_calls) {
      // 1 elemento (4bit x 2, quindi un uint8)
      // Decimate ma senza dotp - Decimo tutto prima
      int8_t inA1 = *pA1++;
      __decimate1_16_block2(vecB1, vecB2, pB1, pB2, maschera_decimate);
      // Decimo gli input
      sum1 += inA1 * vecB1[contatore_shift];
      sum2 += inA1 * vecB2[contatore_shift];
      contatore_shift++;
      leftover_instruction_calls--;
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