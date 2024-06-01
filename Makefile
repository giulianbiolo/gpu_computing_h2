.PHONY: clean build run

build:
	nvcc naive_transpose.cu -o naive
	nvcc block_transpose_conflict.cu -o conflict
	nvcc block_transpose_coalesced.cu -o coalesced

run: build
	sbatch ./sbatch_all.sh

clean:
	rm naive conflict coalesced
