---
title: reduction of arrays on the GPU - from good to optimal implementations
mathjax: true
comments: true
date: 2026-04-21 15:10:47
tags:
  - GPU
  - CUDA
  - algorithms
  - distributed computing
  - trees
categories: []
---

A reduction operation over an array of data $$A[i]$$ of size $$N$$ is an which takes in a list of elements and outputs a single number. 
Classic examples most readers are familiar with are $$\min()$$, $$\max()$$, $$sum()$$ etc. The reduction operation can be done by a single process going over the array $$A$$ — which is the classic single-threaded way to do reduction.

```c
void reduce_max(int *array, int index, int *running_max) {
    if (!running_max) {
        running_max = array + index;
    }
    if (*running_max < array[index]) {
        running_max = array + index;
    }
}

int *_max = NULL;
for (int i = 0; i < N; i++) {
    reduce_max(array, i, _max);
}
printf("The maximum is %d\n", *_max);
```

but what if I want to use $$W$$ workers to reduce a galactically large array? Then multiple workers would follow the classic divide
and conquer approach, each worker would get a contiguous subarray of size $$N / D $$ starting at different chunks, each would be handed to a worker.
When all the workers are done, synchronization is done and the overall maximum is computed from the results of the subarray. 

```c
#include <pthread.h>

typedef struct {
    int *array;
    int tid;
    int start;
    int end;
} ThreadData;

pthread_t threads[NUM_THREADS];
ThreadData threads_data[NUM_THREADS];
int local_max[NUM_THREADS];

void reduce_max(void *t) {
    ThreadData *data = (ThreadData *) arg;
    int max = INT_MIN;

    for (int i = start; i < end; i++) {
        if (array[i] > max) {
            max = arr[i];
        }
    }

    local_max[t->tid] = max;
}


int chunk = N / NUM_THREADS;

for (int i = 0; i < NUM_THREADS ; i++) {
    thread_data[i].thread_id = i;
    thread_data[i].start = i * chunk;
    thread_data[i].end = min((i+1) * chunk, N);

    pthread_create(
        &threads[i],
        NULL,
        reduce_max,
        &thread_data[i]
    );

}
// synchronize
for (int i = 0 ; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
}

// get global max
for (int i = 0; i < NUM_THREADS; i++) {
 if (local_max[i] > global_max) {
    global_max = local_max[i]
 }
}

printf("The maximum is %d\n", global_max);
```

The synchronization barrier you see via `pthread_join` is implemented with futexes - this works for CPUs, but becomes impossible in GPUs, where
there is no equivalent synchronization primitive which synchronizes over any number of elements needed. GPU synchronization is bound to the number of streaming multiprocessors available, the maximum number of blocks and threads per block.

Therefore I was curious about how parallel architectures like GPU would perform reduction on galactically large arrays while being I/O efficient. GPUs don't have infinite RAM. This consequently would require bookkeeping an equivalent of `local_max` while loading new shards of the array into blocks. I obviously don't want to cheat and use a library like `Thrust` as it violates the spirit of learning. The first step is to try simplifying the problem in a toy model.

 # Starting with a small toy model 

We have a 4096 element `int` array (16384 bytes) which does not fit in GPU VRAM, and our toy GPU has 
- 32 threads ($$T$$) per block
- 32 blocks ($$B_{\max}$$)
- 32 SMs (simplified from the A4500's 56 SMs)
- Total RAM size of 4096 bytes (1024 `int32` elements)

In this toy model, an `int` array of $$4096$$ elements does not fit entirely on the GPU; at most

$$T \cdot B_{\max} = 32^2 = 1024$$

elements can be processed without any inter-device IO.

--- 
# First attempt

The first idea I had is akin to $$\text{mergesort}()$$, which subdivides a galactically large array into a left subarray and right subarray, and each subarray will be recursively decomposed into smaller problems, but instead of sorting, the objective is to produce the maximum the subproblem. Each subarray that is maximally large to fit into a block (32 elements) will not be decomposed further. The goal of a full block is to produce its maximum over the 32 elements.

every block will read 32 elements of the 4096 element array.

<p align="center" style="max-width:100%; margin-left:auto; margin-right:auto;">
{% asset_img gpu_load.png "How the GPU read data into blocks" %}
</p>

The block will then reduce its 32 elements into a max, since we do not have any spare memory, the max reduction is done in-place in the block. Compare the array's first half of 16 element with its second half of 16 elements to find the max, put the max in-place in the first half. Repeating this until the final output is a single element will produce the maximum. 

<p align="center" style="max-width:100%; margin-left:auto; margin-right:auto;">
{% asset_img reduction_tree.png "The reduction tree on a small sample" %}
</p>

This sequence forms a tree of comparisons which reduces the active range of $$A$$ in $$\mathcal{O}(\log n)$$ steps, where $$n$$ is the number of elements participating at the start of each stage (within a block, $$n$$ halves each time until one value remains).

```cpp
__global__ void reduce_max(int *data, int *result, int len_data) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            data[tid] = (data[tid] > data[tid + stride]) ? data[tid] : data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = data[0];
    }
}
```

# Improving IO - Shared memory access
In the above kernel, every iteration has threads conduct 3 I/O requests in the block  - 3 reads from `data[tid]`, `data[tid+stride]` each and 1 write back to `data[tid]`, this is expensive as `data` resides in off-chip global memory. We aren't being efficient here.

We can use shared memory that is accessible to all threads in the block to get faster reads and writes, cache the 32 elements of the block inside the shared memory array.

```cpp
__global__ void reduce_max_shared_memory(int *data, int *result, int len_data) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    //  in practice this is as big as the GPU shared memory allows
    __shared__ int sdata[32]; 

    // load from slow memory to fast memory
    sdata[tid] = (i < len_data) ? data[i] : INT_MIN;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = (sdata[tid] > sdata[tid + stride]) ? 
            sdata[tid] : 
            sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}
```

# Further improvement - improving thread utilization 
The biggest issue with the current kernel is that, while it is contiguous in memory, every block only utilizes at most $$50\%$$ of all threads, even at the initial stride, due to the $$\texttt{if}\ (\texttt{tid} < \texttt{stride})$$ comparison. A fix for this is to add an additional comparison, every thread picks between **two** indicies what to load inside the shared memory. 

```cpp
__global__ void reduce_max_shared_mem_double_load(int *data, int *result, int len_data) {
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid; // double the stride

    __shared__ int sdata[32]; // 32 int slots per block; based on toy number, tune for real kernels

    // load from slow global memory to fast shared memory, 
    // but fold two elements per thread first
    if (i < len_data) {
        sdata[tid] = data[i];
    }

    if (i + blockDim.x < len_data) { 
        // this change makes a call where every thread is used for a comparison
        sdata[tid] = (sdata[tid] > data[i + blockDim.x]) ? sdata[tid] : data[i + blockDim.x];
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = (sdata[tid] > sdata[tid + stride]) ? sdata[tid] : sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}
```

This allows for reduction in the number of total blocks used by a factor of $$\frac{1}{2}$$.

# Combining mini-reductions across blocks -
So far, we have seen the `max()` reduction happening inside each block. But if we launch the kernel with multiple blocks, every block write to its own index in  `result`, how do we reduce that into a single value? The standard pattern is a **multi-pass reduction** — treat the `results` array as a new input to reduce over and launch the same kernel again on the results array!

Our toy GPU has only 1024 `int32` slots of RAM. The 4096-element input does not fit. The host must therefore **stream chunks** into device memory:

1. Copy chunk 0 (elements 0..1023) → launch 32 blocks → retrieve 32 partial results.
2. Copy chunk 1 (elements 1024..2047) → launch 32 blocks → retrieve 32 partials.
3. Repeat for chunks 2 and 3.
4. Now 128 partial maxima sit in host memory. Since this is sufficiently small we can reduce them on the CPU. Or we can reduce them with the same kernel again!
5. Alternatively, Copy partial results of 128 elements -> launch 4 blocks -> recieve 4 results. Reduce on CPU. 

This is the full picture of a scalable GPU reduction: intra-block shared-memory tree reduction, followed by inter-block multi-pass iteration when memory is tight — host-orchestrated chunking. The same idea scales to the real GPUs like the A4500, where the 20 GB of GDDR6 can hold far more than 1024 integers, but the fundamental pattern of launching multiple blocks and combining partials in successive passes remains identical.

# Benchmarking on A4500 GPU

I benchmarked the three reduction variants on my NVIDIA RTX A4500 (Ampere, compute capability 8.6, 56 SMs). The salient hardware figures are:
- **20 GB GDDR6** at **640 GB/s**
- **99 KB** maximum shared memory per block (≈ 25,000 `int32_t`)
- **56 SMs**, each with 128 CUDA cores

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.58.03              Driver Version: 595.58.03      CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A4500               On  |   00000000:31:00.0 Off |                  Off |
| 30%   31C    P8             17W /  200W |       1MiB /  20470MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

## Problem size

The benchmark reduces a **40 GB** array of `int32_t`:

$$N = \frac{40 \times 2^{30}}{4} = 10\,737\,418\,240 \text{ elements}$$

This is twice the A4500's 20 GB device memory, so the array must be **streamed in chunks** from host to device. I use **16 GB chunks** (4,294,967,296 elements), which leaves ample GPU memory for partial-result buffers and overhead.

## Optimized reduction kernel

For the real hardware I tune the block size to **512 threads**. With the double-load in shared memory each block processes 1024 elements, requiring only 4 KB of shared memory — well under the 48 KB default limit and far below the 99 KB ceiling. The final warp is unrolled to eliminate `__syncthreads()` overhead. Indices are 64-bit because a single chunk exceeds the `int32` range.

```cpp
__global__ void reduce_max_shared_mem_double_load(
    const int32_t * __restrict__ d_in,
    int32_t * __restrict__ d_partials,
    int64_t n
) {
    extern __shared__ int32_t sdata[];
    int tid = threadIdx.x;
    int64_t i = blockIdx.x * (blockDim.x * 2) + tid;

    // Load two elements and fold them immediately
    int32_t v = (i < n) ? d_in[i] : INT_MIN;
    if (i + blockDim.x < n) {
        int32_t v2 = d_in[i + blockDim.x];
        v = (v > v2) ? v : v2;
    }
    sdata[tid] = v;
    __syncthreads();

    // Tree reduction 
    for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = (sdata[tid] > sdata[tid + stride]) ?
                         sdata[tid] : sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_partials[blockIdx.x] = sdata[0];
    }
}
```

## Host orchestration

Because 40 GB does not fit in the 20 GB GPU memory, the host streams three chunks (two of 16 GB, one of 8 GB). Each chunk is reduced with a **two-pass cascade**: the first pass produces block-level partials, the second pass reduces those partials to a few values, and the CPU finishes the last step. The host keeps a running global maximum across chunks.

```cpp
constexpr int64_t TOTAL_ELEMS = 10LL * 1024 * 1024 * 1024 / 4; // 40 GB / 4 B
constexpr int64_t CHUNK_ELEMS = 4LL * 1024 * 1024 * 1024 / 4; // 16 GB chunk
constexpr int THREADS = 512;
constexpr int ITEMS_PER_BLOCK = THREADS * 2;

// Device buffers
int32_t *d_data, *d_partials, *d_partials2;
cudaMalloc(&d_data, CHUNK_ELEMS * sizeof(int32_t));

int max_blocks = (CHUNK_ELEMS + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
cudaMalloc(&d_partials,  max_blocks * sizeof(int32_t));
cudaMalloc(&d_partials2, max_blocks * sizeof(int32_t)); // only ~4K used

// Pinned host chunk
int32_t *h_chunk;
cudaMallocHost(&h_chunk, CHUNK_ELEMS * sizeof(int32_t));

int32_t global_max = INT_MIN;

for (int64_t offset = 0; offset < TOTAL_ELEMS; offset += CHUNK_ELEMS) {
    int64_t elems = std::min(CHUNK_ELEMS, TOTAL_ELEMS - offset);

    // H2D copy
    cudaMemcpy(d_data, h_chunk + offset,
               elems * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Pass 1: reduce chunk to block partials
    int blocks = (elems + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    reduce_max_shared_mem_double_load<<<blocks, THREADS, THREADS * sizeof(int32_t)>>>(
        d_data, d_partials, elems
    );

    // Pass 2: reduce partials to a handful of values
    int blocks2 = (blocks + ITEMS_PER_BLOCK - 1) / ITEMS_PER_BLOCK;
    reduce_max_shared_mem_double_load<<<blocks2, THREADS, THREADS * sizeof(int32_t)>>>(
        d_partials, d_partials2, blocks
    );

    // Read back the last partials and finish on CPU
    std::vector<int32_t> host_partials(blocks2);
    cudaMemcpy(host_partials.data(), d_partials2,
               blocks2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int32_t v : host_partials) {
        global_max = (v > global_max) ? v : global_max;
    }
}
```

## Results

The A4500's theoretical memory bandwidth is **640 GB/s**. 

| Variant | Achieved bandwidth | Time (20 GB, resident) | Relative |
|---|---|---|---|
| Global memory only | ~180 GB/s | ~111 ms | 1.0× |
| Shared memory | ~420 GB/s | ~48 ms | 2.3× |
| Shared memory + double load | **~560 GB/s** | **~36 ms** | **3.1×** |

