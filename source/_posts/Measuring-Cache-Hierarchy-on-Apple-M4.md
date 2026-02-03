---
title: Measuring Cache Hierarchy on Apple M4 with Pointer Chasing
mathjax: true
comments: true
date: 2026-01-31
tags:
  - Apple Silicon
  - M4
  - Cache
  - Systems Programming
  - Performance
  - Memory Hierarchy

---

While exploring parallel computing concepts on my own time, I decided to set up an experiment to measure cache latencies on my M4 Max MacBook Pro. 
The results revealed some interesting characteristics of Apple Silicon,particularly the 128-byte cache lines and the dramatic latency differences across the memory hierarchy.

This post walks through the pointer chasing technique I used, the M4-specific findings, and what the numbers mean for writing cache-friendly code.

---

## Key Takeaways

**What you'll learn:**
1. **What is the memory hierarchy and what benefits does it confer?**
2. **How to measure cache sizes** using pointer chasing,a technique that defeats CPU prefetching
3. **What are your CPU's actual latency numbers:** My M4 Max CPU had a L1 latency (~0.7ns) → L2 (~1.5ns) → Main memory (~6.4ns)
4. **Why M4's 128-byte cache lines are different**,double Intel's 64-byte lines, affects struct padding and [false sharing](https://en.wikipedia.org/wiki/False_sharing)
5. **Practical [working set](https://en.wikipedia.org/wiki/Working_set) guidelines:** Keep hot data under 128KB (L1) or 16MB (L2) when possible

# What is the [memory hierarchy](https://en.wikipedia.org/wiki/Memory_hierarchy)?
<p align="center">
  <img src="/2026/01/31/Measuring-Cache-Hierarchy-on-Apple-M4/RAM_hierarchy.jpeg" style="max-width:100%; height:auto;">
</p>

<p align="center" style="margin-top: 10px; font-style: italic; color: #666;">
  <strong>M4 Cache Hierarchy:</strong> Performance cores feature 128 KB L1 and access to a shared 16 MB L2, with latencies ranging from ~0.7 ns (L1) to ~6.4 ns (main memory).
</p>

If you are a layperson unfamiliar with computer architecture here is an overly simplified description of memory hierarchy. Similar to humans, the CPU has levels of memory: short-term memory to remember things that are important fleetingly then quickly disposed and long-term memory to hold older memories which are further buried and take longer to "surface". The short-term memory is the fastest, but it also holds the least amount of information and is overwritten often, while the long-term memory is further and slower to access but is rarely overwritten due to how large it is.

The fastest level is the **[L1 cache](https://en.wikipedia.org/wiki/CPU_cache#L1_cache)**, then the **[L2 cache](https://en.wikipedia.org/wiki/CPU_cache#L2_cache)** and finally the **[RAM](https://en.wikipedia.org/wiki/Random-access_memory)**. The smallest unit of memory you can read and write to the cache is the **[cache line](https://en.wikipedia.org/wiki/CPU_cache#Cache_entry_structure)**.


**Why does this matter?**

AI assistance based *vibe coding* and *vibe engineering* has dramatically decreased the barrier needed to architect a program and has dramatically increased the number of people using code they indirectly designed (knowingly or unknowingly), however even cutting-edge AI based coding assistants need to be explicitly told to prioritize performance. Whether a person is writing it themselves, or if they are telling AI,not knowing the underlying memory hierarchy will put you at a disadvantage, especially in more constrained situations such as when running on microcontrollers, Arduinos or Raspberry Pi.

As we'll see below, understanding the memory hierarchy can reveal a **9x latency difference**,cache-optimized code can be dramatically faster than cache-oblivious code.

## Checking Your Cache Configuration

Before running any benchmarks, you can query your system's cache parameters directly. On macOS, the `sysctl` command gives this information: 

```bash
# Cache line size
sysctl hw.cachelinesize
# hw.cachelinesize: 128 (bytes)

# L1 data cache size (P-core)
sysctl hw.perflevel0.l1dcachesize
# hw.perflevel0.l1dcachesize: 131072 (128 KB)

# L2 cache size (P-core)
sysctl hw.perflevel0.l2cachesize
# hw.perflevel0.l2cachesize: 16777216 (16 MB)
```

Note that this is just a simplification, computers usually have multiple cores and the cache size will actually vary on core specs depending on which core handles the query. For example M4 Max CPUs have 14 cores,10 *performance* cores and 4 *efficiency* cores,and the efficiency cores have a 64 KB L1 and 4 MB L2, while the performance cores have 128 KB L1 and access the shared 16 MB L2.

These values give us targets to validate with our experiments,if the pointer chasing technique works correctly, we should see latency transitions at these boundaries.


## Cache Benefits: Cache-Aware vs Cache-Oblivious Code

To illustrate the practical impact of cache hierarchy, consider matrix transpose,a common operation for linear algebra. 

If you ask the latest and greatest coding language model **Claude Opus 4.5-20251101** with 74.4% on SWEBench (link to https://www.swebench.com/) to output a matrix transpose function on row major matrices, you get this:

**Cache-Oblivious Approach - Opus 4.5 20251101**

```c
void transpose_naive(double *src, double *dst, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[j * n + i] = src[i * n + j];  // Column-major write
        }
    }
}
```

**Problem:** The inner loop writes to `dst` with stride `n`, jumping across cache lines. For a 1024×1024 matrix of doubles (8 bytes each), each write is 8192 bytes apart,far exceeding M4's 128-byte cache line. The place we read from at index src[i, j] only pulled at most the next 15 doubles from src, and that range does not include the place we wish to write to, meaning the code is forced to do another read from RAM into the L1 cache!

<div id="cache-animation-container"></div>
<script src="/2026/01/31/Measuring-Cache-Hierarchy-on-Apple-M4/cache-animation.js"></script>

**Cache-Aware (Blocked) Approach:**

```c
// Do the transpose block by block, taking tiles over the large input matrix and output matrix
void transpose_blocked(double *src, double *dst, int n, int block_size) {
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            // Get the block
            int i_max = (i + block_size < n) ? i + block_size : n;
            int j_max = (j + block_size < n) ? j + block_size : n;
            // Transpose one block at a time
            for (int bi = i; bi < i_max; bi++) {
                for (int bj = j; bj < j_max; bj++) {
                    dst[bj * n + bi] = src[bi * n + bj];
                }
            }
        }
    }
}
```

**Why blocking helps:**

- Processes small tiles (e.g., 64×64) that fit in L1 cache, each cache line pulled is reused multiple times.
- Each block is fully transposed before moving to the next
- Reduces cache misses by maximizing reuse of loaded cache lines
- On M4 with 128 KB L1, a 64×64 double matrix (32 KB) fits comfortably

<div id="blocked-animation-container"></div>
<script src="/2026/01/31/Measuring-Cache-Hierarchy-on-Apple-M4/cache-animation-blocked.js"></script>

**Practical implications:**
- Hot data structures or memory (the working set) under 128 KB benefit from L1 speeds
- Working sets under 16 MB stay in L2 territory
- Random access across large arrays incurs the full main memory penalty


Now, I have hopefully convinced you that cache and cacheline awareness matters.

---

# Benchmarking latency at different Cache levels on M4 Max

## Why Naive Benchmarks Don't Work

The problem with measuring memory latency directly is a first world problem, computers are very good now at predicting what you are going to do next and preemptively pulling that data into cache via [prefetching](https://en.wikipedia.org/wiki/Cache_prefetching):

- Modern CPUs use hardware prefetching to anticipate access patterns
- Sequential or strided access is highly predictable and triggers prefetching, masking true latency
- Need a technique where each access depends on the previous result

This is where pointer chasing comes in,it creates a data dependency that the prefetcher cannot anticipate.


## How Pointer Chasing Works

Each memory access depends on the *value* returned by the previous access. The CPU cannot prefetch because the next address is unknown until the current load completes.

**Setting up the pointer chain:**

```c
// Create a circular chain where each element points stride_size ahead
void initialize_pointer_chain(uintptr_t *memory, uint64_t num_elements, uint32_t stride_size) {
    for (uint64_t i = 0; i < num_elements; i++) {
        uint64_t next = (i + stride_size) % num_elements;
        memory[i] = (uintptr_t)(&memory[next]);
    }
}
```

**The traversal loop:**

```c
// Chase pointers,each load depends on the previous, defeating prefetching
// We use an arbitrarily large iteration count (1 billion) to average out noise
volatile uintptr_t *current = &memory[0];
for (uint64_t i = 0; i < 1000000000ULL; i++) {
    current = (volatile uintptr_t *)(*current);
}
```

The `volatile` keyword prevents the compiler from optimizing away the repeated loads. Without it, the compiler may hoist the load out of the loop or eliminate it entirely, invalidating the benchmark.

---

## Apple Silicon Differences

Key platform constants for M4:

```c
#ifdef __APPLE__
    #define CACHE_LINE_SIZE 128   // vs 64 bytes on Intel
    #define L1_CACHE_SIZE (128 * 1024)  // 128 KB per P-core
    #define L2_CACHE_SIZE (16 * 1024 * 1024)  // 16 MB shared
    #define PAGE_SIZE (16 * 1024)  // 16 KB vs 4 KB on Linux
#endif
```

**M4 Cache Hierarchy:**

| Level | Performance Cores | Efficiency Cores |
|-------|------------------|------------------|
| L1 | 128 KB | 64 KB |
| L2 | 16 MB (shared) | 4 MB (shared) |
| L3 | None | None |
| Memory | Unified | Unified |

**Key differences from Intel/AMD:**
- 128-byte cache lines (double the typical 64 bytes)
- No L3 cache,relies on large L2 and unified memory architecture
- 16 KB page size vs 4 KB on most Linux systems

**Practical implications:**
- Struct padding for cache alignment needs 128-byte boundaries on M4
- False sharing thresholds differ,threads writing to addresses within 128 bytes may contend
- Memory alignment functions like `aligned_alloc` should use 128-byte alignment

---

## Experiment 1: Measuring Cache Sizes

Using the pointer chasing technique described above, we can measure cache sizes by varying the buffer size we traverse. When the buffer fits entirely in a cache level, all accesses hit that cache and we observe its characteristic latency. When the buffer exceeds a cache's capacity, accesses spill to the next level and latency increases.

We allocate buffers ranging from 4 KB to 32 MB (powers of 2), set up a pointer chain in each, and measure the average time per access. The stride is fixed at 128 bytes,one cache line per access,so each pointer chase touches a new cache line rather than benefiting from spatial locality within a line.

**Results:**

<p align="center">
  <img src="/2026/01/31/Measuring-Cache-Hierarchy-on-Apple-M4/cache_latency_plot.png" style="max-width:100%; height:auto;">
</p>

<p align="center" style="margin-top: 10px; font-style: italic; color: #666;">
  <strong>Figure 1:</strong> Access latency vs. buffer size. Clear plateaus reveal cache boundaries at 128 KB (L1) and 16 MB (L2).
</p>

**Analysis:**

The plot shows three distinct regions:
1. **Flat region (up to ~128 KB):** L1 cache,consistent low latency around 0.7 ns
2. **Gradual increase (128 KB to 16 MB):** L2 cache,latency rises to ~1.7 ns
3. **Sharp jump (beyond 16 MB):** Main memory,latency spikes to ~6.4 ns

The boundaries align with M4's documented cache sizes: 128 KB L1 (performance cores) and 16 MB L2.

**Takeaway:** Working set size directly determines which latency tier your code operates in. The difference between fitting in L2 vs spilling to main memory is a 4-9x latency penalty.

---

## Experiment 2: Measuring Cache Line Size

To measure cache line size, we fix several buffer sizes and vary the stride from 8 bytes to 256 bytes. When the stride is smaller than the cache line, consecutive accesses may hit the same line,but once the stride exceeds the cache line size, every access fetches a new line and latency increases.

**Results:**

<p align="center">
  <img src="/2026/01/31/Measuring-Cache-Hierarchy-on-Apple-M4/cache_line_plot.png" style="max-width:100%; height:auto;">
</p>

<p align="center" style="margin-top: 10px; font-style: italic; color: #666;">
  <strong>Figure 2:</strong> Latency vs. stride for various buffer sizes. The transition at 128 bytes confirms M4's cache line size.
</p>

**Analysis:**

The stride vs. latency plot reveals a consistent pattern across buffer sizes:
- Strides ≤128 bytes: Similar latency (accessing within single cache line or adjacent lines)
- Strides >128 bytes: Latency increases (each access touches a new cache line)

The transition at 128 bytes confirms M4's cache line size. For larger buffer sizes (those that exceed L1), the effect is more pronounced because cache misses occur more frequently.

---

## Practical Guidelines

**Working set sizing:**
- Under 128 KB: L1 speeds (~0.7 ns/access)
- Under 16 MB: L2 speeds (~1.5 ns/access)
- Over 16 MB: Main memory (~6.4 ns/access)

**Data layout considerations:**

How you organize your structs matters. A struct larger than 128 bytes spans multiple cache lines, meaning accessing any field may trigger multiple cache fetches. Separating frequently-accessed "hot" fields from rarely-accessed "cold" fields keeps the hot path cache-friendly.

```c
// Large struct spans multiple cache lines
struct LargeRecord { char data[256]; };  // 2 cache line fetches

// Separating hot and cold data
struct HotFields { int id; float value; };  // Fits in one line
struct ColdFields { char metadata[200]; };  // Accessed less frequently
```

**Access patterns:**

The prefetcher is your friend for predictable access,sequential or strided patterns get detected and data arrives in cache before you need it. Random access across large arrays defeats prefetching entirely, paying the full memory latency on every access.

---

## Implementation Notes

A few issues encountered during implementation:

- **No `pthread_barrier_t` on macOS:** Had to implement barrier synchronization manually for multi-threaded bandwidth tests
- **Compiler optimization:** The `volatile` keyword is essential; without it, `-O3` will eliminate the pointer chase loop entirely
- **Timer resolution:** 1 billion iterations ensures timing measurements exceed timer granularity and average out system noise
- **Memory alignment:** Used `aligned_alloc()` with 128-byte alignment for consistent results

---

## Conclusion

The memory hierarchy has a direct impact on performance. On M4:
- L1 to main memory represents a ~9x latency difference
- 128-byte cache lines differ from Intel's 64-byte lines
- Working set size determines which cache level dominates your access patterns

The pointer chasing technique provides a way to measure these characteristics empirically, useful for validating assumptions and getting a deep understanding of your own hardware.
