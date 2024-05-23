cmake_minimum_required(VERSION 3.19.5)

# Application

set(TARGET_NAME         "main")
set(MAIN                "main.c")


# Cmake pre-inits


set(CONFIG_GAP_SDK_HOME    $ENV{GAP_SDK_HOME})
include ($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
set(BOARD_NAME             gap9_evk)
set(PMSIS_OS               freertos)
set(platform               $ENV{PLATFORM})

set (CORE                    8)
set (GAP_SDK                 1)
set (TARGET_CHIP_FAMILY_GAP9 1)


# set (FLASH_TYPE DEFAULT)
# set (RAM_TYPE   DEFAULT)
# set (MODEL_L3_FLASH   AT_MEM_L3_DEFAULTFLASH)
# set (EXEC_FROM_FLASH true)
# set (MODEL_L3_RAM   AT_MEM_L3_DEFAULTRAM)

set (FREQ_CL 370)
set (FREQ_FC 370)
set (FREQ_SFU 370)
# set (VOLTAGE 650)
set (VOLTAGE 800)
set (CLUSTER_STACK_SIZE 8192)
set (CLUSTER_SLAVE_STACK_SIZE 4096)
set (TOTAL_STACK_SIZE $(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* $(CLUSTER_NUM_CORES)))
set (MODEL_L1_MEMORY $(shell expr 120000 \- $(TOTAL_STACK_SIZE)))
set (MODEL_L2_MEMORY 1000000)
set (MODEL_L3_MEMORY 8000000)





set (BUILD_DIR            ${CMAKE_CURRENT_SOURCE_DIR}/build)
set (DORY_DIR             ${CMAKE_CURRENT_SOURCE_DIR}/dory)
set (STANDALONE_CRT_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/runtime)
set (CODEGEN_PATH         ${CMAKE_CURRENT_SOURCE_DIR}/codegen)
set (GAP9_SDK_DIR         $ENV{GAP_SDK_HOME})

list (APPEND TARGET_INCS  
                         -I${GAP9_SDK_DIR}/libs/gap_lib/include
                         -I${GAP9_SDK_DIR}/rtos/pmsis/bsp/include/bsp
                         -I${GAP9_SDK_DIR}/rtos/pmsis/bsp/include/bsp/ram
                         -I${GAP9_SDK_DIR}/rtos/pmsis/bsp/include/bsp/flash
                         -I${GAP9_SDK_DIR}/rtos/sfu/include
                         -I.
                         -I${CMAKE_CURRENT_SOURCE_DIR}/include
			 -I${DORY_DIR}/include
			 -I${STANDALONE_CRT_PATH}/src/runtime/crt/include
			 -I${STANDALONE_CRT_PATH}/include
                         -I${CODEGEN_PATH}/host/include
)

list (APPEND TARGET_SRCS  
			  ${STANDALONE_CRT_PATH}/src/runtime/crt/common/crt_backend_api.c
                          ${STANDALONE_CRT_PATH}/src/runtime/crt/memory/stack_allocator.c
                          
)   




FILE (GLOB CODEGEN_C_Sources ${CODEGEN_PATH}/host/src/*.c)
list (APPEND TARGET_SRCS ${CODEGEN_C_Sources})

FILE (GLOB DORY_C_Sources ${DORY_DIR}/src/*.c)
list (APPEND TARGET_SRCS ${DORY_C_Sources})

FILE (GLOB MATCH_C_Sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
list (APPEND TARGET_SRCS ${MATCH_C_Sources})



list(APPEND TARGET_CFLAGS   -O2
                            -s
                            -w
                            -g
                            -mno-memcpy
                            -fno-tree-loop-distribute-patterns
                            -Wno-incompatible-pointer-types
                            # -Dabort\(\)=exit\(\-\1\)
                            -Wno-error=format
)



list(APPEND TARGET_PREPROCESSOR

			    -DNUM_CORES=${CORE}
			    -DNDEBUG
                            -DPULP
                            -DGAP_SDK=${GAP_SDK}
                            -DFLASH_TYPE=${FLASH_TYPE}
                            -DUSE_${FLASH_TYPE}
                            -DUSE_${RAM_TYPE}
                            -DSTACK_SIZE=${CLUSTER_STACK_SIZE}
                            -DSLAVE_STACK_SIZE=${CLUSTER_SLAVE_STACK_SIZE}
                            -DFREQ_FC=${FREQ_FC}
                            -DFREQ_CL=${FREQ_CL}
                            -DFREQ_SFU=${FREQ_SFU}
                            -DVOLTAGE=${VOLTAGE}
			    -DSINGLE_CORE_DMA
			    -DGAP_SDK=${GAP_SDK}
			    -DTARGET_CHIP_FAMILY_GAP9

)


project (${TARGET_NAME} C ASM)

add_executable (${TARGET_NAME} ${TARGET_SRCS})

target_compile_options(${TARGET_NAME} PUBLIC   ${TARGET_CFLAGS}
                                               ${TARGET_INCS}
                                               ${TARGET_PREPROCESSOR}

)
target_link_libraries(${TARGET_NAME} PRIVATE m)
target_link_libraries(${TARGET_NAME} PRIVATE ${TILER_LIB})
list(APPEND TABLE_CFLAGS -lm)

# CMake post initialization

setupos(${TARGET_NAME})

