# Benchmarks for validating the perf changes. See ../README.md.
#
#   make                      build everything that has its dependencies
#   make uvm_tlb_bench        needs nvcc (CUDA toolkit)
#   make nvml_rpc_bench       needs nvml.h and libnvidia-ml.so
#   make uvm_page_tree_test   plain C
#
# CUDA_HOME defaults to /usr/local/cuda. NVML_INC can point at a directory
# containing nvml.h if it is not under $(CUDA_HOME)/include.

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
NVML_INC  ?= $(CUDA_HOME)/include
CC        ?= cc
CFLAGS    ?= -O2 -Wall -Wextra
NVCCFLAGS ?= -O2 -std=c++17

all: uvm_page_tree_test nvml_rpc_bench uvm_tlb_bench

uvm_tlb_bench: uvm_tlb_bench.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

nvml_rpc_bench: nvml_rpc_bench.c
	$(CC) $(CFLAGS) -I$(NVML_INC) -o $@ $< -lnvidia-ml

uvm_page_tree_test: uvm_page_tree_test.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f uvm_tlb_bench nvml_rpc_bench uvm_page_tree_test

.PHONY: all clean
