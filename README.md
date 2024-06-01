# GPU Computing - Deliverable 2

## File Structure:
```
|-naive_transpose.cu            # Cuda Naive Transpose Implementation
|-block_transpose_conflict.cu   # Cuda Block Transpose Implementation with shared memory
|-block_transpose_coalesced.cu  # Cuda Block Transpose Implementation with shared memory and coalesced memory access
|-gen_report.py                 # Python script to generate the graphs from the output data of the cuda programs
```
## Usage:

First of all enter via SSH to the cluster and clone the repository.
Then, in the repository folder, run the following commands:

```bash
module load cuda       # Load the cuda module
make run               # This will build the executables with nvcc and schedule a sbatch job to run all of them
```
At this point we can copy the generated data from the `transpose-*.out` file to our local machine.
Then pass the filename of the output data to the `gen_report.py` script to generate the graphs:
```
python gen_report.py <filename>
```
This will generate a `graphs` folder in the current directory, containing the various graphs generated from the output data you have provided.
