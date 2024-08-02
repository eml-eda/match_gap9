CORE ?= 8
#gdbserver = 1
#gdbport=3333
platform=gvsoc
BUILD_DIR := .
DORY_DIR := dory
SPARSE_DIR := sparse
# Include paths
STANDALONE_CRT_PATH := $(abspath $(BUILD_DIR))/runtime
CODEGEN_PATH := $(abspath $(BUILD_DIR))/codegen

# Sources
STANDALONE_CRT_SRCS := $(STANDALONE_CRT_PATH)/src/runtime/crt/common/crt_backend_api.c
STANDALONE_CRT_SRCS += $(STANDALONE_CRT_PATH)/src/runtime/crt/memory/stack_allocator.c

CODEGEN_SRCS = $(wildcard $(abspath $(BUILD_DIR))/codegen/host/src/*.c)
CODEGEN_OBJS = $(subst .c,.o,$(CODEGEN_SRCS))

DORY_SRCS = $(wildcard $(abspath $(DORY_DIR))/src/*.c)
DORY_OBJS = $(subst .c,.o,$(DORY_SRCS))
DORY_INC_PATH = $(abspath $(DORY_DIR))/include

SPARSE_SRCS = $(wildcard $(abspath $(SPARSE_DIR))/src/*.c)
SPARSE_OBJS = $(subst .c,.o,$(SPARSE_SRCS))
SPARSE_INC_PATH = $(abspath $(SPARSE_DIR))/include
#def zigzag

MATCH_SOURCES = $(wildcard src/*.c)

INC_PATHS =  -I${STANDALONE_CRT_PATH}/include \
			 -I${STANDALONE_CRT_PATH}/src/runtime/crt/include \
			 -I${CODEGEN_PATH}/host/include \
			 -I ./include/ \
			 -I${DORY_INC_PATH}\
			 -I${SPARSE_INC_PATH}\

# Defining the source files for pulp-runtime
APP = main
APP_SRCS = $(STANDALONE_CRT_SRCS) $(CODEGEN_SRCS) $(DORY_SRCS) $(SPARSE_SRCS)
APP_SRCS += $(MATCH_SOURCES)
PULP_INC_PATHS += -DSDK

# Note that this value can automatically be changed by python scripts:
OPT_LEVEL = 3
# Use -DNDEBUG to remove assertions in TVM runtime
# Use -DPULP for preprocessor in malloc wrapper
# Need to link in libm for math.h inclusion in softmax operator
APP_CFLAGS += -DNUM_CORES=$(CORE) -g $(INC_PATHS) -DNDEBUG -O$(OPT_LEVEL) -DPULP 
# GAP9 fixes, the first because abort() is not in stdlib. The second to avoid a crt_backend_api.c error
APP_CFLAGS += -Dabort\(\)=exit\(\-\1\)  -Wno-error=format -w
#-fno-indirect-inlining -flto -w
APP_LDFLAGS += -lm 
#-Wl,--print-memory-usage -flto



GAP9_DEFAULT_FLASH_TYPE = DEFAULT_FLASH
GAP9_DEFAULT_RAM_TYPE = DEFAULT_RAM

GAP8_DEFAULT_FLASH_TYPE = HYPERFLASH
GAP8_DEFAULT_RAM_TYPE = HYPERRAM

PULP_DEFAULT_FLASH_TYPE = HYPERFLASH
PULP_DEFAULT_RAM_TYPE = HYPERRAM

FLASH_TYPE ?= $($(TARGET_CHIP_FAMILY)_DEFAULT_FLASH_TYPE)
RAM_TYPE ?= $($(TARGET_CHIP_FAMILY)_DEFAULT_RAM_TYPE)

ifeq '$(FLASH_TYPE)' 'MRAM'
READFS_FLASH = target/chip/soc/mram
endif

APP_CFLAGS += -DFLASH_TYPE=$(FLASH_TYPE) -DUSE_$(FLASH_TYPE) -DUSE_$(RAM_TYPE)
APP_CFLAGS += -DSINGLE_CORE_DMA


APP_CFLAGS += -DGAP_SDK=1
APP_CFLAGS += -DTARGET_CHIP_FAMILY_$(TARGET_CHIP_FAMILY)

################################### SPECIAL ####################################

# The special rule modifiers (.PHONY etc...) go here

#################################### RULES #####################################

include $(RULES_DIR)/pmsis_rules.mk

##################################### EOF ######################################
