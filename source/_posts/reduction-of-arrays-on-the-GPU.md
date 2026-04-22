---
title: reduction of arrays on the GPU - from good implementations to optimal implementations
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

A reduction operation over an array of data $$A[i]$$ of size $$N$$ is an operation like $$\min()$$, $$\max()$$, $$\sum()$$, etc.;
the reduction operation can be done by a single worker processing the array $$A$$ — which is the classic
single-threaded way to do reduction.

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

but I was curious about how it would be possible to the equivalent of this reduction in CUDA device without
excessive synchronization, how would a block be able to keep a tally of the running maximum without excessive synchronization between blocks? 

Let's start with a toy model which ignores the specifics of any GPU.

# Toy model

$$N = 1024,\qquad B_{\max} = 32,\qquad T = 32$$

Here $$N$$ is the problem size, $$B_{\max}$$ is the maximum number of blocks, and $$T$$ is the threads per block (taken equal to warp size / SM “width” in this sketch). The array $$A$$ is contiguous in memory.

In this toy model, an `int` array of $$4096$$ elements does not fit entirely on-chip; at most

$$T \cdot B_{\max} = 32^2 = 1024$$

elements can be processed without cross-device memory reads.

The first idea I had is akin to $$\text{mergesort}()$$, which subdivides each array into subarrays. Each subarray is recurisively decomposed into smaller problems, but instead of sorting, the objective is to produce the maximum the subproblem. This is done inside each block. 
---- 

We can process $$32$$ indices in $$A$$ concurrently with $$32$$ streaming multiprocessors.
Inside a block, thread $$t$$ can use the global index

$$
i = \texttt{block\_idx} \cdot \texttt{block\_size} + t
$$

and compare with the thread at offset $$\texttt{block\_size >> 1} = 16$$ (half the block away in thread index) to produce partial $$\max$$ values, still keyed by the same index formula for the active half.
Once done, we can drop the latter half of the array by marking those threads inactive.
```text
time step 1
<block 0 : threads 0 - 31>
    <block 0 : tid 0 - global index  0 >  compare with  <block  0 : tid 16> and update tid 0
    <block 0 : tid 1 - global index  1 >  compare with  <block  0 : tid 17> and update tid 1
    <block 0 : tid 2 - global index  2 >  compare with  <block  0 : tid 18> and update tid 2
    <block 0 : tid 3 - global index  3 >  compare with  <block  0 : tid 19> and update tid 3
    <block 0 : tid 4 - global index  4 >  compare with  <block  0 : tid 20> and update tid 4
    <block 0 : tid 5 - global index  5 >  compare with  <block  0 : tid 21> and update tid 5
    <block 0 : tid 6 - global index  6 >  compare with  <block  0 : tid 22> and update tid 6
    <block 0 : tid 7 - global index  7 >  compare with  <block  0 : tid 23> and update tid 7
    <block 0 : tid 8 - global index  8 >  compare with  <block  0 : tid 24> and update tid 8
    <block 0 : tid 9 - global index  9 >  compare with  <block  0 : tid 25> and update tid 9
    <block 0 : tid 10 - global index 10 >  compare with <block  0 : tid 26> and update tid 10
    <block 0 : tid 11 - global index 11 >  compare with <block  0 : tid 27> and update tid 11
    <block 0 : tid 12 - global index 12 >  compare with <block  0 : tid 28> and update tid 12
    <block 0 : tid 13 - global index 13 >  compare with <block  0 : tid 29> and update tid 13
    <block 0 : tid 14 - global index 14 >  compare with <block  0 : tid 30> and update tid 14
    <block 0 : tid 15 - global index 15 >  compare with <block  0 : tid 31> and update tid 15

time step 2
<block 0 : threads 0 - 31>
    <block 0 : tid 0 - global index  0 >  compare with  <block  0 : tid 8> and update tid 0
    <block 0 : tid 1 - global index  1 >  compare with  <block  0 : tid 9> and update tid 1
    <block 0 : tid 2 - global index  2 >  compare with  <block  0 : tid 10> and update tid 2
    <block 0 : tid 3 - global index  3 >  compare with  <block  0 : tid 11> and update tid 3
    <block 0 : tid 4 - global index  4 >  compare with  <block  0 : tid 12> and update tid 4
    <block 0 : tid 5 - global index  5 >  compare with  <block  0 : tid 13> and update tid 5
    <block 0 : tid 6 - global index  6 >  compare with  <block  0 : tid 14> and update tid 6
    <block 0 : tid 7 - global index  7 >  compare with  <block  0 : tid 15> and update tid 7
```
-----

<p align="center" style="max-width:100%; margin-left:auto; margin-right:auto;">
{% asset_img reduction_tree.png "The reduction tree on a small sample" %}
</p>

This sequence forms a tree of comparisons which reduces the active range of $$A$$ in $$\mathcal{O}(\log n)$$ steps, where $$n$$ is the number of elements participating at the start of each stage (within a block, $$n$$ halves each time until one value remains).

```text
Initial:   [1, 2, 3, 4, 5, 6, 7, 8]
stride=4:  [5, 6, 7, 8, X, X, X, X]   (tid 0-3 active)
stride=2:  [7, 8, X, X, X, X, X, X]   (tid 0-1 active)
stride=1:  [8, X, X, X, X, X, X, X]   (tid 0 active)
```

```cpp
__global__ void reduce_max(int *data, int *result, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
        if (tid < stride) {
            data[tid] = (data[tid] > data[tid + stride]) ? data[tid] : data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = data[tid];
    }
}
```

# Shared memory access
In the above kernel, every iteration has threads conduct 5 expensive I/O requests - 2 reads from `data[tid]`, `data[tid+stride]` each and 1 write back to `data[tid]`, this is expensive as it accessing off chip global memory only available via. DDR / HBM I/O calls. We can use shared memory that is shared across all threads in the block to get faster reads and writes across time steps.

```cpp
__global__ void reduce_max_shared_memory(int *data, int *result, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    __shared__ int sdata[32]; // 32 int slots per block; tune for real kernels

    // load from slow memory to fast memory
    sdata[tid] = (i < n) ? data[i] : 0;
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
        *result = sdata[tid];
    }
}
```

The biggest issue with the current kernel is that, while it is contiguous in memory, every block only utilizes at most $$50\%$$ of all threads, even at the initial stride, due to the $$\texttt{if}\ (\texttt{tid} < \texttt{stride})$$ comparison. A fix for this is to assign each initial thread **two** initial indices instead of one index. 

```cpp
__global__ void reduce_max_shared_memory(int *data, int *result, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + tid;

    __shared__ int sdata[32]; // 32 int slots per block; tune for real kernels

    // load from slow memory to fast memory, but fold two elements per thread first
    if (i < n) {
        sdata[tid] = data[i];
    }

    if (i + blockDim.x < n) {
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
        *result = sdata[tid];
    }
}
```

This allows for reduction in the number of blocks used by a factor of $$\frac{1}{2}$$.

