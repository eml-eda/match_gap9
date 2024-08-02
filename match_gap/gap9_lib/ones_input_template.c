#include <stdio.h>
#include <stdint.h>
#include "tvmgen_default.h"
#include <tvm_runtime.h>
#include <malloc_wrapper.h>
#include <gdb_anchor.h>
#define MAX_PRINT_ERRORS 128
#define STOP_AT_FIRST_ERROR 0
int abs(int v) {return v * ((v > 0) - (v < 0)); }
% if compare_with_correct:
uint8_t expected_result[${match_output["size"]}]=${expected_results};
% endif
int main(int argc, char** argv) {
  % if target!="x86":
  gap9_cluster_init();
  % endif

  uint32_t output_size = ${match_output["size"]};
  uint8_t *output = (uint8_t*)malloc_wrapper(output_size * sizeof(uint8_t));
  struct tvmgen_default_outputs outputs = { .output = output, };
  
  % for match_input in match_inputs:
  uint32_t ${match_input["name"]}_size = ${match_input["size"]};
  uint8_t *${match_input["name"]} = (uint8_t*)malloc_wrapper(${match_input["name"]}_size * sizeof(uint8_t));
  // Fill input with 1
  for (uint32_t i = 0; i < ${match_input["name"]}_size; i++){
    ${match_input["name"]}[i] = 1;
  }
  % endfor

  struct tvmgen_default_inputs inputs = {
    % for idx,match_input in enumerate(match_inputs):
    % if idx>0:
    ,
    % endif
    .${match_input["name"]} = ${match_input["name"]}
    % endfor
  };

  int32_t status = 0;
  printf("\n{");
  printf("\"kernel_cycles\":[0");
  //start_g_perf_counter();
  status = tvmgen_default_run(&inputs, &outputs);
  //stop_g_perf_counter();
  //int32_t cycles=get_acc_perf_counter();
  int errors=0;
  //printf("],\"cycles\":%d,",cycles);
  % if compare_with_correct:
  printf("\"errors\":[");
  for(int k=0;k<output_size;k++){
    if(expected_result[k]!=((uint8_t)output[k])){
      errors++;
      if(STOP_AT_FIRST_ERROR)
        break;
      else{
        if(errors<MAX_PRINT_ERRORS){
          if(errors>1) printf(",");
          printf("[%d,%d,%d]",k,expected_result[k],output[k]);
        }
      }
    }
  }
  printf("],");
  % endif
  printf("\"num_errors\":%d,",errors);
  if(!errors) printf("\"correct\":true,");
  else  printf("\"correct\":false,");
  printf("\"output\":[");
  % if log_output:
  for(int k=0;k<output_size;k++) {printf("%d",(uint8_t)output[k]);if(k!=output_size-1) printf(", ");}
  % else:
  //for(int k=0;k<output_size;k++) {printf("%d",(uint8_t)output[k]);if(k!=output_size-1) printf(", ");}
  % endif
  printf("]");
  printf("}\n");
  % for match_input in match_inputs:
  free_wrapper(${match_input["name"]});
  % endfor
  free_wrapper(output);
  % if target!="x86":
  gap9_cluster_close();
  % endif
  if(status != 0){
    abort();
  }
  return 0;
}
