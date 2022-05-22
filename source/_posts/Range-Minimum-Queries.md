---
title: Range Minimum Queries and Approaches
mathjax: true
comments: true
date: 2022-05-19 21:10:15
tags: 
- Algorithms
- Range Minimum Queries
- Trees
---

A common problem in computer science is to query for a property over a range of contiguous elements in a given set.

Examples of this problem would be
- **For array `$A$` find the minimum element between `$\forall i \le j \; A[i \ldots j]$`  **

The difficulty in the problem is that there are `$O(n^2)$` contiguous subsets of an array.

# Approach 1 - Naive Brute Force Approach

```python
def compute_minimum_over_range(A: List):
    n = len(A)
    range_min_query = [[0] * n ] * n 

    for i in range(n):
        for j in range(i, n):
            minimum = A[i]
            for k in range(i, j+1):
                minimum = min(minimum, A[k])
            
            range_min_query[i][j] = minimum
    
    return range_min_query
```

The naive method iterates over every possible subarray of `$A$` and computes the minimum over the subarray. Since computing minimum over an array is linear time operation we compute:

`$\sum_{k=1}^{n} k(n-k+1) = \sum_{k=1}^{n} k(n+1)-k^2 = (n+1)\sum_{k=1}^{n} k - \sum_{k=1}^{n} k^2$`

which is in `$O(n^3)$`


# Approach 2 - Dynamic Programming

In the naive, brute force approach we are repeating computations which could be used for speeding up queries for larger ranges. 

A simple example would be for 
```
A = [1, 5, -1, 3, 4]
```
a smart approach would be to notice that `$A[0 \ldots 2]$` has smaller subproblems `$A[0 \ldots 1]$` and `$A[1 \ldots 2]$` and we can get the range_min_queryult in constant time from subproblems when we know them, like observing that:

`$\min(A[0 \ldots 2]) = \min(A[0 \ldots 1], A[1 \ldots 2])$`

instead of repeating the computation
`$\min(A[0 \ldots 2]) = \min(A[0], A[1], A[2])$`

so if we save range_min_queryults to the subproblems in memory, the future superproblems will encounter can be sped up.


```python
def compute_minimum_over_range(A: List):
    # use dynamic programming
    n = len(A)
    range_min_query = [[0] * n] * n

    # start from smaller subproblems to larger ones
    for subproblem_size in range(1, n+1):
        for i in range(n):
            j = i + subproblem_size - 1

            if subproblem_size == 1:
                # base case, minimum of one element is itself 
                range_min_query[i][j] = A[i]
            else:
                range_min_query[i][j] = min(range_min_query[i][j-1], range_min_query[i+1][j])
    
    return range_min_query
```

At the cost of using `$O(n^2)$` space we have sped up our algorithm to `$O(n^2)$` time.

A query for a specific range `$i \ldots j$` we can get the query range_min_queryult in constant time by performing a look up on the range_min_queryult of `compute_minimum_over_range` dynamic programming table.
# Approach 3 - Using Powers of Two optimization

A optimization of the above dynamic programming approach is to exploit the fact that the `$\min$` operation over sets does not care about ovelaps.

For any set `$A$`, `$\min(A \cup B) = \min(A)$` where `$B \subseteq A$`.

We can optimize by not computing our range minimum query table `$RMQ$` for every subset size, instead using a sensible scheme such as powers of two. This can be a recurrence relation `$A[i\ldots i+2^k-1]$` where `$i$` is the position and `$k$` is the power of two we want to query over.

`$ RMQ_A(i, k) = \left\{\begin{array}{ll} A[i] & k == 0 \\ RMQ(i,k-1) & RMQ(i,k-1) \le RMQ(i+2^{k-1}, k-1) \\ RMQ(i+2^{k-1}, k-1) & RMQ(i,k-1) \gt RMQ(i+2^{k-1}, k-1)\end{array}\right\}$`



```python
def compute_minimum_over_range(A: List):
    n = len(A)
    K = ceil(log2(n))
    range_min_query = [[0] * n ] * K 

    for k in range(K):
        for i in range(n):
            if k == 0:
                range_min_query[k][i] = A[i]
                continue

            L = range_min_query[k-1][i]
            U = range_min_query[k-1][i + 1<<(k-1)]

            range_min_query[k][i] = min(L, U)

    return range_min_query
```

This reduces the range minimum query computation table to `$O(n\log(n))$`

# Further Approaches

It is possible to do better than `$O(n\log(n))$` time and `$O(1)$` lookup time for range minimum queries. In fact, it is possible to do `$O(n)$` preprocessing and `$O(1)$` lookup through multiple clever data structures I will discuss in the future. 
