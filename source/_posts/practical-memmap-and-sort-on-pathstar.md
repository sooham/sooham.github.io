---
title: What Transformers Actually Memorize — Geometric Learning on PathStar Graphs
mathjax: true
comments: true
date: 2025-11-09 15:12:46
tags:
    - NLP
    - Sequence Modelling
    - Next Token Prediction
    - PathStar
    - Geometric Learning
    - Loss Functions
---

When a language model memorizes a fact like "Paris is the capital of France," what exactly is it storing? The standard answer is *associative memory*: the model learns a lookup table in its weight matrices, pairing co-occurring tokens. But a 2025 paper by Noroozizadeh et al. presents a striking counter-narrative: deep sequence models tend to memorize *geometrically*, arranging their token embeddings to encode global structural relationships that were never explicitly supervised.

I spent several months replicating and extending their experiments on a task called **InWeightPathStar**. Along the way I learned that getting the loss function right was far more important than any architectural decision, and that the geometry that emerges in embedding space is genuinely beautiful. This post walks through what the PathStar task is, why it should be impossibly hard for transformers, what actually happens when you train one, and the critical loss function insight that made it all work.

---

## What is a PathStar Graph?

A PathStar graph is a tree with a deceptively simple structure. There is a single **root** node at the center, and $d$ **spokes** radiating outward, each of length $l$. Every spoke is a chain of nodes from the root to a leaf. The total graph has $d \times (l - 1) + 1$ vertices and $d \times (l - 1)$ edges.

<!-- TODO: Diagram 1 — PathStar graph with d=5, l=4. Show root at center, 5 colored spokes radiating outward, each with 3 edges and a labeled leaf at the tip. Label root, intermediate nodes, and leaves. Use distinct colors per spoke. -->

For example, with $d = 100$ spokes and $l = 5$ nodes per spoke, we get a graph with 401 vertices and 400 edges. The topology is trivial for a human to understand, but it turns out to be adversarially constructed against next-token prediction models.

### Why PathStar is Hard: The In-Context Failure

The PathStar task was originally designed by Bachmann and Nagarajan (2024) as an *in-context* reasoning challenge. In the in-context version, a model receives a randomized adjacency list of a fresh PathStar graph in its prompt, along with a start node (the root) and a goal node (a leaf). The model must output the unique path from root to goal.

Next-token trained transformers fail spectacularly at this. The failure unfolds in two stages:

1. **The Clever Hans cheat.** During training with teacher forcing, the model discovers a shortcut: for every token *except the first*, it can simply predict the unique neighbor of the previous ground-truth token that was revealed in the context. This left-to-right cheat is far easier to learn than the actual right-to-left planning needed to trace a path from root to leaf through the adjacency list.

2. **Gradient starvation.** Because the cheat works for all tokens except the first, the loss gradients are dominated by the easy tokens. The hard first token — which requires composing $l$ lookups through the adjacency list — never receives enough gradient signal to learn. At test time, the model guesses the first token randomly and then faithfully follows the wrong spoke.

This is a *needle in a haystack* problem: the model must search through $\exp(l)$ possible compositions to find the correct first hop, but the loss landscape is flat everywhere except at the correct answer.

### The In-Weights Variant: Memorize the Graph Into the Weights

Noroozizadeh et al. proposed a crucial twist: instead of giving the graph in-context, **make the model memorize the entire graph structure into its weights** through training. This is the **InWeightPathStar** task, and it is the focus of this post.

The setup uses a single fixed PathStar graph. Training involves two interleaved tasks:

**Edge memorization.** The model sees individual edges and learns to predict neighbors:

```
[EDGE, u, GT, v]    — "from node u, going away from root (GT), reach node v"
[EDGE, v, LT, u]    — "from node v, going toward root (LT), reach node u"
```

Only the final token (the predicted neighbor) contributes to the loss. The first tokens are context.

**Path finding.** The model sees a leaf node and must produce the entire path from root to that leaf:

```
[PATH, leaf, PAUSE, ..., PAUSE, root, n₁, n₂, ..., leaf]
```

The PAUSE tokens give the model computational slack — extra positions to "think" before it must commit to the first prediction (the root's child on the correct spoke). Loss is computed only on the path tokens after the pauses.

<!-- TODO: Diagram 2 — Annotated sequence diagram showing one EDGE example and one PATH example side by side. Show the token sequence, with colored boxes for context tokens (no loss) and prediction tokens (loss computed). Show the loss mask as 0s and 1s below each token. -->

A fraction of the leaves (typically 20%) are held out for validation. The model is trained on edges from the *entire* graph plus paths from the training leaves, and evaluated on its ability to produce correct paths for the *held-out* leaves it has never seen as path targets.

The remarkable finding: **the model succeeds.** On graphs with up to $5 \times 10^4$ nodes, a decoder-only transformer trained from scratch achieves near-perfect accuracy on held-out paths. It has somehow learned to do multi-hop implicit reasoning purely from its memorized edge knowledge, without ever being shown the held-out paths during training.

---

## Teacher Forcing: How Training Works

The model is a standard decoder-only transformer (GPT architecture) trained with **teacher forcing**. During training, at each position in the sequence, the model receives the *ground-truth* previous token as input and predicts the next token. The loss is cross-entropy between the model's prediction distribution and the actual next token.

This is the standard regime for autoregressive language models, but there is a subtle gap: during inference, the model must use its *own* predictions as input (since ground truth is not available). If the model makes an error at position $t$, all subsequent predictions may be corrupted.

### Bridging the Gap: Scheduled Sampling

To mitigate the train-test mismatch, the codebase implements **scheduled sampling** (also called autoregressive substitution). With probability $p_{\text{sub}}$ at each position, the ground-truth input token is replaced with the model's own greedy prediction from the previous step.

```python
for pos in range(path_context_length, seq_len):
    with torch.no_grad():
        logits_partial, _ = model(X_modified[:, :pos], targets=None)
    predicted = logits_partial[:, -1, :].argmax(dim=-1)

    should_substitute = (torch.rand(batch_size) < p_sub) & is_path
    X_modified[should_substitute, pos] = predicted[should_substitute]

    # Mask the target at pos-1 since the model predicted its own input
    Y_modified[should_substitute, pos - 1] = -1
```

When a token is substituted, the loss for *predicting* that substituted token is masked (the target would be the model's own prediction, which is meaningless), but the loss for predicting the *next* token given the substituted input is kept. This forces the model to learn to recover from its own errors.

In practice, scheduled sampling makes training slower (each position requires a separate forward pass) but produces models that are more robust during autoregressive inference. EDGE tasks always use pure teacher forcing since they are single-step predictions.

### Interleaving and Class Balancing

A practical challenge: the number of edges in a PathStar graph far exceeds the number of paths. With $d = 100$ and $l = 5$, there are 800 edges (400 directed $\times$ 2 for undirected) but only 80 training paths. If we naively interleave them, the model sees edges $10\times$ more often than paths.

The fix is **deterministic tiling**: the path dataset is replicated (tiled) to match the edge dataset size, ensuring 50/50 representation in each epoch. This is not random oversampling — it is exact repetition, which guarantees every path gets equal training time.

```
Edges: 800 samples
Paths: 80 samples → tiled to 800 samples
Interleaved: 1600 samples per epoch, 50% edges, 50% paths
```

---

## The Loss Function Journey

This section describes the most important lesson from the entire project. The loss function evolved through several stages, and getting it right was the difference between a model that plateaued at mediocre accuracy and one that converged to near-perfect performance.

### Phase 1: Naive Cross-Entropy

The first implementation used standard cross-entropy loss on all predicted tokens:

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1,
    reduction='mean'
)
```

This treats every prediction as a single-answer classification problem: there is exactly one correct next token, and the model is penalized for assigning probability to anything else.

For PATH tasks, this is correct — at every position in a spoke, there is exactly one correct next node. But for EDGE tasks, there is a fundamental problem.

### The Root Node Problem

Consider the root node. It has $d$ children, one per spoke. When the model sees:

```
[EDGE, root, GT, ???]
```

it must predict which child comes next. But *every* child of the root is a valid answer — the edge `(root, child_i)` exists for all $i \in \{1, \ldots, d\}$.

With standard cross-entropy, each training example picks one specific child as the "correct" answer, and the model is penalized for assigning probability to the other $d - 1$ equally valid children. The gradients from different training examples fight each other: one example says "child 1 is correct, punish everything else," the next says "child 2 is correct, punish everything else."

The model can never reach zero loss on these examples. The **theoretical minimum loss** is bounded by the irreducible entropy at the root:

$$L_{\min} = \frac{d \cdot \ln(d)}{\text{total tokens per epoch}}$$

This insight came from writing a dedicated analysis script. For $d = 100$ and $l = 5$, the root contributes 100 edge training examples, each with $\ln(100) \approx 4.605$ nats of irreducible entropy. The rest of the graph is deterministic — every non-root node has exactly one parent and at most one child per direction.

### Phase 2: Adaptive Loss with KL Divergence

The fix is conceptually simple: instead of treating edge prediction as single-answer classification, treat it as matching a *distribution* over valid neighbors. For a source node $u$ with neighbor set $\mathcal{N}(u)$, the target distribution is uniform over the neighbors:

$$P_{\text{target}}(v \mid u) = \begin{cases} \frac{1}{|\mathcal{N}(u)|} & \text{if } v \in \mathcal{N}(u) \\ 0 & \text{otherwise} \end{cases}$$

The loss for edge tasks becomes the KL divergence between this uniform target and the model's predicted distribution:

$$L_{\text{edge}} = D_{KL}(P_{\text{target}} \| Q_{\text{model}}) = \ln\!\left(\frac{1}{|\mathcal{N}(u)|}\right) - \frac{1}{|\mathcal{N}(u)|} \sum_{v \in \mathcal{N}(u)} \ln Q_{\text{model}}(v)$$

For the root node ($|\mathcal{N}| = d$), the optimal model output is $\frac{1}{d}$ probability for each child — and this achieves *zero* KL divergence. The model is no longer penalized for the inherent ambiguity.

For non-root nodes that have only 1–2 neighbors, the KL loss reduces to something very close to standard cross-entropy, so the model still learns precise predictions where precision is possible.

### Implementation: Precomputed Neighborhood Buffers

The KL divergence requires knowing each node's neighbor count at training time. This is precomputed once during model initialization and stored as GPU-resident buffers:

```python
def _precompute_neighborhood_info(self):
    vocab_size = self.config.vocab_size
    max_neighbors = max(len(n) for n in adj_list.values())

    neighborhood_tensor = torch.full(
        (vocab_size, max_neighbors), -1, dtype=torch.long
    )
    neighborhood_sizes = torch.zeros(vocab_size, dtype=torch.long)

    for node, neighbors in adj_list.items():
        n = sorted(list(neighbors))
        neighborhood_sizes[node] = len(n)
        neighborhood_tensor[node, :len(n)] = torch.tensor(n)

    inv_sizes = torch.zeros(vocab_size, dtype=torch.float32)
    mask = neighborhood_sizes > 0
    inv_sizes[mask] = 1.0 / neighborhood_sizes[mask].float()

    self.register_buffer('neighborhood_tensor', neighborhood_tensor)
    self.register_buffer('neighborhood_sizes_tensor', neighborhood_sizes)
    self.register_buffer('inv_neighborhood_sizes_tensor', inv_sizes)
```

The forward pass then dispatches to different loss functions based on the task token at position 0:

```python
# Identify EDGE vs PATH tasks
is_edge = (idx[:, 0] == EDGE_token)
is_path = (idx[:, 0] == PATH_token)

# KL divergence for edges
if is_edge.any():
    source_nodes = idx[edge_indices, 1]
    neighborhoods = self.neighborhood_tensor[source_nodes]
    inv_sizes = self.inv_neighborhood_sizes_tensor[source_nodes]

    Q = F.softmax(edge_logits, dim=1)
    log_q = torch.log(Q[batch_idx, neighborhoods] + 1e-10)
    log_q_masked = log_q.masked_fill(~valid_mask, 0.0)

    kl = torch.log(inv_sizes) - inv_sizes * log_q_masked.sum(dim=1)
    total_loss += kl.sum()

# Standard cross-entropy for paths
if is_path.any():
    path_loss = F.cross_entropy(
        path_logits.reshape(-1, path_logits.size(-1)),
        path_targets.reshape(-1),
        ignore_index=-1, reduction='sum'
    )
    total_loss += path_loss
```

### The Accuracy Metric Also Had to Change

The loss fix required a corresponding change in how we measure edge prediction accuracy. With standard cross-entropy, accuracy is simply "did the model's argmax match the single target?" But with a uniform target over $|\mathcal{N}(u)|$ neighbors, the right question is: "do the model's top-$|\mathcal{N}(u)|$ predictions cover the actual neighbor set?"

```
For root (d=100 neighbors):
  Old metric: "Did the model predict child_42?" → ~1% accuracy even for a perfect model
  New metric: "Do the top-100 predictions cover all 100 children?" → 100% for a perfect model

For non-root (1-2 neighbors):
  Both metrics agree — top-1 or top-2 should cover the neighbor set.
```

This neighborhood-based accuracy metric reflects what the model actually needs to learn: which nodes are neighbors, not which single neighbor is "most correct."

---

## What Emerges in Embedding Space

After training converges, the learned token embeddings tell a remarkable story. Using UMAP to project the high-dimensional embeddings into 2D or 3D, we can visualize the structure the model has discovered.

### Geometric Structure by Depth

When we color each node by its depth (distance from root), a clear gradient emerges in embedding space. Nodes at the same depth cluster together, even though they belong to completely different spokes that share no training examples.

<!-- TODO: Diagram 3 — 2D UMAP projection of learned embeddings, colored by depth in the PathStar graph. Root at center, depth-1 nodes in one color band, depth-2 in another, etc. Leaves should form the outermost cluster. Use a sequential colormap (e.g., viridis) for depth. Generate from a trained checkpoint using visualize_embeddings_umap.py with --color-by depth. -->

### Similarity to Root Decays with Distance

The cosine similarity between each node's embedding and the root embedding decreases smoothly with graph distance. This is not supervised — the model was never told about the global structure of the graph. It only saw individual edges and individual paths.

<!-- TODO: Diagram 4 — Line plot: x-axis is distance from root (0 to l-1), y-axis is cosine similarity to root embedding. Show min/avg/max bands. Should show a smooth, monotonically decreasing curve. Generate from a trained checkpoint using the summary figure from visualize_embeddings_umap.py. -->

This smooth gradient is the signature of **geometric memory**: the embeddings encode a global notion of distance that was inferred from local co-occurrence information. Under the associative memory hypothesis, embeddings would be essentially random relative to each other, and this gradient would not exist.

### Path-Specific Structure

The most compelling evidence for geometric memory comes from comparing *within-path* and *cross-path* similarity. For a given leaf node, we measure the cosine similarity between the leaf's embedding and every other node's embedding.

Within the leaf's own spoke, similarity is high and decays smoothly toward the root. Across other spokes, similarity is lower overall, with a pattern driven primarily by depth rather than path identity.

<!-- TODO: Diagram 5 — Side-by-side plots. Left: "Within-Path Similarity" — for 5 selected paths (5 colors), plot cosine similarity between the leaf and each node along the same path, x-axis is distance from root. Right: "Cross-Path Similarity" — average cosine similarity between each leaf and nodes on OTHER paths, by depth. Within-path should be visibly higher than cross-path. Generate from a trained checkpoint using the leaf similarity comparison from visualize_embeddings_umap.py. -->

When within-path similarity significantly exceeds cross-path similarity, it means the model has learned more than just "nodes at depth 3 are similar." It has learned which specific depth-3 node belongs to which spoke — a global structural property that emerges from purely local edge supervision.

### Watching Geometry Emerge During Training

The codebase supports generating UMAP snapshots at regular intervals during training and stitching them into animated GIFs. The progression is striking:

- **Early training:** Embeddings are scattered randomly in high-dimensional space, reflecting the random initialization.
- **Mid-training:** Rough clustering by depth begins to appear. The root separates from everything else. Leaves start grouping together.
- **Late training:** Clear geometric structure aligned with the graph topology. Spokes become visible as coherent trajectories in embedding space.

<!-- TODO: Diagram 6 — Embedding evolution GIF or 3-panel figure showing UMAP projections at early (epoch ~100), mid (epoch ~5000), and late (epoch ~15000) training stages. Color by path/spoke identity. Use 5-6 highlighted paths with distinct colors, rest in gray. Generate using the embedding GIF pipeline from model.py's create_embedding_gif_from_checkpoints(). -->

---

## Practical Lessons

### The Loss Function Matters More Than Architecture

I spent weeks tuning model architecture — number of layers, embedding dimension, attention heads, MLP width — with marginal improvements. Switching the edge loss from cross-entropy to KL divergence over the neighbor distribution produced an immediate, dramatic improvement. The model went from plateauing at mediocre path accuracy to converging toward near-perfect accuracy.

The lesson: when your task has *inherent ambiguity* at certain positions (like the root of a PathStar graph), a loss function that penalizes that ambiguity will fight against itself. Aligning the loss with the actual structure of the problem is worth more than any amount of hyperparameter search.

### Always Compute the Theoretical Minimum

Before debugging a model that "won't converge," compute what the optimal loss *should* be. For PathStar, the theoretical minimum is:

$$L_{\min} = \frac{d \cdot \ln(d)}{\text{total tokens per epoch}}$$

If your training loss is approaching this bound, the model is doing as well as theoretically possible and your remaining loss is irreducible entropy. If it is far above, you have a real optimization problem. If it goes *below* this bound, you have a bug in your loss computation or masking.

### NaN Debugging: Detect Before Backward

Large embedding dimensions ($n_{\text{embd}} \geq 256$) without regularization are prone to gradient explosion. The chain: unbounded parameters $\to$ large activations $\to$ numerical overflow in softmax $\to$ NaN in backward pass. The fix is twofold:

1. **Early detection**: Check for NaN in the loss and logits *before* calling `backward()`. By the time `backward()` crashes, it is too late for diagnostics.

2. **Prevention**: Always use gradient clipping (`grad_clip=1.0`) with large embeddings. Weight decay and dropout help but are not sufficient alone.

```python
# Check BEFORE backward
if torch.isnan(loss) or torch.isinf(loss):
    raise ValueError("NaN detected in loss — stopping before corruption")

scaler.scale(loss).backward()  # Safe: forward pass was clean
```

### Balance Your Interleaved Datasets

When mixing tasks with very different dataset sizes (800 edges vs. 80 paths), the minority task gets drowned out. Deterministic tiling (not random oversampling) ensures exact balance and reproducibility.

---

## Conclusion

The PathStar task provides a clean sandbox for studying how sequence models store knowledge. The in-context version reveals a fundamental failure mode of next-token prediction: gradient starvation of the hardest decision. The in-weights version shows that transformers can overcome this limitation by developing *geometric* embeddings that encode global graph structure from local supervision.

The most important practical takeaway is about loss function design. When your data has positions with inherent ambiguity — where multiple answers are equally valid — a loss function that acknowledges this ambiguity (like KL divergence against a uniform distribution over valid answers) can unlock performance that no amount of standard cross-entropy optimization will reach.

The geometry that emerges is not just a curiosity. As Noroozizadeh et al. argue, if we can understand and strengthen this geometric bias, it could improve implicit reasoning, knowledge retrieval, and generalization in practical language models. The gap between the Transformer's geometry and the cleaner geometry of Node2Vec models suggests significant headroom remains in the embedding architectures we use today.

---

## References

1. Noroozizadeh, S., Nagarajan, V., Rosenfeld, E., & Kumar, S. (2025). *Deep sequence models tend to memorize geometrically; it is unclear why.* arXiv:2510.26745. [https://arxiv.org/abs/2510.26745](https://arxiv.org/abs/2510.26745)

2. Bachmann, G. & Nagarajan, V. (2024). *The Pitfalls of Next-Token Prediction.* ICML 2024. [https://arxiv.org/abs/2403.13112](https://arxiv.org/abs/2403.13112)

3. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)

4. Elhage, N., Hume, T., Olsson, C., et al. (2022). *Toy Models of Superposition.* Transformer Circuits Thread, Anthropic. [https://transformer-circuits.pub/2022/toy_model/index.html](https://transformer-circuits.pub/2022/toy_model/index.html)
