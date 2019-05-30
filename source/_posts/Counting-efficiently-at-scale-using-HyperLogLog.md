---
title: Counting efficiently at scale using HyperLogLog
mathjax: true
comments: true
date: 2019-05-28 19:28:02
categories:
- Algorithms 
tags: 
- HyperLogLog
---
Counting **exactly** number of distinct elements from a stream at scale is intrinsically a linear problem. From a perspective of time taken, intuition tells us that we cannot skip any elements in the stream and be confident in the accuracy of our result. With regards to space required to solve the problem, a record of previously seen elements in the stream must be kept. For a stream of `$n$` distinct elements, the **COUNT-DISTINCT** algorithm requires `$\Theta(n)$` space.

For services counting distincts in huge streams, such as the number of IP addresses visiting [Google](https://google.com/), the number of unique views of a popular Reddit post, the memory-to-utility ratio would not justify using **COUNT-DISTINCT**, instead developers can use an approximation algorithm of **COUNT-DISTINCT** to produce a estimate close enough at a fraction of the space needed. **HyperLogLog** is one such approximation algorithm.


### Implementation of COUNT-DISTINCT


```cpp
#include <map>

using namespace std;

template <class stream_type>
class COUNT_DISTINCT {
    std::map<stream_type, int> seen;
    
    public:
        COUNT_DISTINCT();
        void add(stream_type v);
        size_t count();
}

template<class stream_type>
COUNT_DISTINCT<stream_type>::COUNT_DISTINCT() {}

template<class stream_type>
void COUNT_DISTINCT<stream_type>::add(stream_type v) {
    seen[v] = 1;
}

template<class stream_type>
void COUNT_DISTINCT<stream_type>::count() {
    return seen.size();
}
```

## What is HyperLogLog Exactly?


HyperLogLog is many things, from the perspective of computer science, it is a:

- **Streaming algorithm** - an algorithm operating on read-once-only stream of data.
- **Approximation algorithm** - an algorithm which can be mathematically proven to produce a result within a bound of the optimal result.
- **A probabilistic algorithm** - an algorithm not guaranteed to be correct for every input, in fact you'll see that HyperLogLog can be passed a stream by an adversary to make it fail by a large margin. However, if the input stream is randomly distributed has a sufficiently large `$n$`, as in most applications at scale, HyperLogLog will behave well.
- **A data structure**: While HyperLogLog was described as a cardinality estimation algorithm by Flajolet et. al. We can segment the logic of this algorithm into methods to create a data structure consuming elements in a stream and providing an estimation.

The main principle behind HyperLogLog is understanding the effect of random hashing a stream of values. Given a hash function `$hash(x): D \rightarrow \{0,1\}^L$` mapping the stream domain to a string of `$L$` bits uniformly, with every bit being a independent and identically distributed random variable. Hashing to random will create a new probability distribution which is indicative of the distinct count.

If we `$hash$` a stream of random integers between `$[0, 10^9]$`, it is expected that 50% of hashed elements in the stream start with bit pattern `$1$`, 25% of the hashed elements start with `$01$`, 12.5% start with `$001$`, 6.25% start with `$0001$`. Intuitively it makes sense that seeing the bit pattern `$0^\rho1$` means `$n$` is at least `$2^{\rho}$`.

A simple version of HyperLogLog keeps track of the random variable `$r = \max({msb(hash(x_i)) : \forall i \in S})$` over the stream `$S$`, where `$msb$` is the index of the first non-zero most significant bit, i.e `$msb(0001_2) = 3$`, `$msb(1000_2) = 0$`, `$msb(0100_2) = 1$` and `$msb(0000_2) = 4$` over 4-bit inputs. The value of `$2^{r}$` is returned, with `$r = -\infty$` when the stream is empty.

## A simple, but naive version of HyperLogLog
```cpp
#include <cmath>
#include <functional>

unsigned int msb_index(size_t i) {
    // return the index of the left most significant bit of input i
    if (i == 0) return sizeof(size_t) * 8;
    unsigned int n = 0;
    size_t nth_msb_bitmask = ((size_t) 1) << (sizeof(size_t) * 8 - 1);
    while((i & nth_msb_bitmask) == 0 && n < sizeof(size_t) * 8) {
        n++;
        nth_msb_bitmask = ((size_t) 1) << (sizeof(size_t) * 8 - 1 - n);
    }
    return n;
}


template <class stream_type>
class HyperLogLog {
    size_t r;
    std::hash<stream_type> hash_func;

    public:
    HyperLogLog();
    void add(stream_type v);
    double estimate();

};

template <class stream_type>
HyperLogLog<stream_type>::HyperLogLog(), hash_func()  {
    r = 0;
}

template <class stream_type>
void HyperLogLog<stream_type>::add(stream_type v) {
    size_t hv = hash_func(v);
    r = std::max(r, msb_index(hv));
}

template <class stream_type>
double HyperLogLog<stream_type>::estimate() {
    return pow(2, r);
}
```

The above algorithm is almost identical to a precursor of the HyperLogLog algorithm, the [Flajolet–Martin algorithm](https://en.wikipedia.org/wiki/Flajolet%E2%80%93Martin_algorithm). The above algorithm has a flaw in that it has large variance, if for example the stream contains an element `$x$` for which `$hash(x) = 0$`, then regardless of the number of distinct elements, the result would be `$2^L$`. 

To reduce the flaw, we can partition a stream `$S$` into `$M$` substreams, where `$M < S$`, compute `$r_i$`  for each substream `$i \in \{1, \ldots, M\}$` and average the estimates given by each stream. A simple method to partition the stream is by sorting hashes of elements by their first `$b$` bits. 

Implementing this gets closer to HyperLogLog.

## Implementation of HyperLogLog

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>
#include <algorithm>

unsigned int msb_index(size_t i) {
    // return the index of the left most significant bit of input i
    if (i == 0) return sizeof(size_t) * 8;
    unsigned int n = 0;
    size_t nth_msb_bitmask = ((size_t) 1) << (sizeof(size_t) * 8 - 1);
    while((i & nth_msb_bitmask) == 0 && n < sizeof(size_t) * 8) {
        n++;
        nth_msb_bitmask = ((size_t) 1) << (sizeof(size_t) * 8 - 1 - n);
    }
    return n;
}


template <class stream_type>
class HyperLogLog {
    unsigned int b; // partition into 2^b substreams
    unsigned int m; // m substreams
    unsigned int* registers;
    std::hash<stream_type> hash_func;

    public:
    HyperLogLog(int b);
    ~HyperLogLog() {}

    void add(stream_type v);
    double estimate();

};

template <class stream_type>
HyperLogLog<stream_type>::HyperLogLog(unsigned int b): b(b), hash_func()  {
    m = 1 << b; 
    registers = new unsigned int[m]();
}

template <class stream_type>
HyperLogLog<stream_type>::~HyperLogLog() {
    delete[] registers;
}

template <class stream_type>
void HyperLogLog<stream_type>::add(stream_type v) {
    size_t hv = hash_func(v);
    size_t j = hv >> (sizeof(size_t) * 8 - b);
    size_t w = hv & ((1 << sizeof(size_t) * 8 - b) - 1);
    registers[j] = std::max(registers[j], msb_index(w));
}

template <class stream_type>
double HyperLogLog<stream_type>::estimate() {
    // return the harmonic mean (reciprocals of the mean of the reciprocals) of 2^{register[i]}
    double harmonic_mean = 0;
    for (int i=0; i < m; i++) {
        harmonic_mean += pow(2.0, -registers[i]);
    }
    harmonic_mean = pow(harmonic_mean, -1.0) * m;
    // harmonic mean computed
    return harmonic_mean * m;
}
```
The intuition behind this algorithm, as written by Flajolet et. al is: 

> Let `$n$` be the unknown cardinality of stream `$S$`. Each substream will comprise approximately `$n/m$` elements. Then, its max-parameter should be close to `$\log_2(n/m)$`. The harmonic mean `$k$`  of the quantities `$2^{Max}$` is then likely to be of the order of `$n/m$`. Thus, `$mk$` should be of the order of `$n$`. 

In the above code for **HyperLogLog::estimate** the final result is a biased estimator, hence the authors multiply the estimate with a constant to remove the bias. The reasoning behind the value chosen is justified by complex math, hence why I have decided not to include the constant in this code.

## Why does HyperLogLog work?
A thorough analysis of HyperLogLog is a complicated issue, requiring good knowledge of Complex analysis and statistical techniques such as Depossionisation. Mathematically adept readers [can read the paper by Flajolet et. al](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf). I may revisit the topic when I have gained mathematical maturity.

## Where is HyperLogLog used?
- In databases such as Redis, Presto and other SQL query engines.
- Used by Reddit to provide a real-time estimate of reddit post counts [[Link](https://redditblog.com/2017/05/24/view-counting-at-reddit/)].



## References
https://code.fb.com/data-infrastructure/hyperloglog/

https://storage.googleapis.com/pub-tools-public-publication-data/pdf/40671.pdf

http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf

https://redditblog.com/2017/05/24/view-counting-at-reddit/