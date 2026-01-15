---
title: What next token prediction models can't do well in-context.
mathjax: true
comments: true
date: 2025-11-09 15:12:46
tags:
    - NLP
    - Sequence Modelling
    - Next Token Prediction
    - Pathstar
---

# High level plan:
- Explain the architecture of next token prediction models i.e transformer based models
    - p(x_t | x_{t-1}, x_{t-2}, ... x_{0})
    - how do gradients propagate in next token prediction models across time
    - how this leads to "needle in a haystack" problem
- Explain pathstar task and objective with in context learning 
    - describe an example pathstar task with D spokes of length L, in context
    - describe the backward task predict the backward path through spoke given the leaf
        - show that this works well
    - describe the forward task predict the forward path through spoke given the leaf   
        - show that this results in a needle in a haystack problem
- What are the workarounds for this needle in a haystack problem
    - different curriculum to learn, learn to memorize the model a la. word2vec, show previous work on node2vec
    - how embeddings in a transformers residual space break down on pathstar tasks. 
- How node2vec learns more than just associations, it learns a global geometry of the graph.


############# Node2vec form ############
E is embedding matrix (d=dim,  n=vocab_size) which is column vectors [e_1 e_2 ... e_n]
[e_11 e_21 ... e_n1]
[e_21 e_22 ... e_n2]
...
[e_1d e_2d ... e_nd]

E^T is unembedding matrix (tied weights) (n=vocab_size, d=dim)
Let y_hat_one_hot(u) = E^T E u, where u is a one hot column vector of shape (n, 1) that is 1 at index x and 0 elsewhere
and the loss L_softmax_one_hot(u) is the cross entropy loss of y_hat_one_hot(u) and u which is one hot on index x
in terms of the column vectors, e_x is embedding at column x of shape (d, 1), thus, we can write y_hat_one_hot(u) as
y_hat_one_hot(u) = y_hat(x)
y_hat(x) = E^T e_x
y_hat(x) = [row(e_1^T) row(e_2^T) ... row(e_n^T)] e_x 
y_hat(x) = [e_1^T e_2^T ... e_n^T] e_x
y_hat(x) = (e_i^T e_x) for i in [1, n]
y_hat(x) = (y_hat_i(x)) for i=1...n (vector notation)
and y_hat_i(x) = (\sum_{j=1}^{d} e_ji e_jx)
\sum_{j=1}^{d} e_jx e_jx

thus for one hot vector u x 
L_softmax_one_hot(u) = -log(cross_entropy(y_hat_one_hot(u), u))
L_softmax_one_hot(u) = -log(cross_entropy(y_hat(x), u))
L_softmax_one_hot(u) = -log(exp(y_hat_x(x)) / \sum_{i=1}^{n} exp(y_hat_i(x)))
L_softmax_one_hot(u) = -log(exp(\sum_{j=1}^{d} e_jx e_jx) / \sum_{i=1}^{n} exp(\sum_{j=1}^{d} e_ji e_jx))

which is a quadratic form look at the expansion of x^T W x where W is a matrix of shape (k, k)
x^T W x = \sum_{i=1}^{k} \sum_{j=1}^{k} x_i w_ij x_j
 
#################### More math on e^T W_assoc e quadratic forms, where e is embedding ##################
See https://claude.ai/chat/06081abb-868a-4056-a9d6-58a5966ae59f
Key insights:
- if 0 is an eigenvalue of W, then W is not invertible , determinant is 0
- if 0 is not an eigenvalue of W, then W is invertible, determinant is non zero


# Sources
1. Noroozizadeh, S., Nagarajan, V., Rosenfeld, E., & Kumar, S. (2025).
Deep sequence models tend to memorize geometrically; it is unclear why.
arXiv:2510.26745. https://arxiv.org/abs/2510.26745 

2. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781. https://arxiv.org/abs/1301.3781
