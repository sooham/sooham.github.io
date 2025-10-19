"""
Stochastic Variational Inference for Mixture of Gaussians
Based on the equations from the blog post on Variational Inference
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, logsumexp
from scipy.stats import dirichlet, norm
import seaborn as sns
from scipy.special import gammaln

# Set random seed for reproducibility
np.random.seed(42)

# Generate data from mixture of Gaussians
means = [30, 50, 50, 100]
stds = [np.sqrt(20), np.sqrt(20), np.sqrt(100), np.sqrt(20)]
weights = [0.5, 0.35, 0.03, 0.12]
sigma_squared = 20  # Known variance

# Sample from the mixture of Gaussians
n_samples = 10000
component_samples = np.random.choice(len(means), size=n_samples, p=weights)
samples = np.array([np.random.normal(means[i], stds[i]) for i in component_samples])

print(f"Generated {n_samples} samples from mixture of Gaussians")
print(f"True means: {means}")
print(f"True weights: {weights}")
print(f"Sample mean: {samples.mean():.2f}, Sample std: {samples.std():.2f}")

# Create visualization of the mixture
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot histogram of samples
ax.hist(samples, bins=50, density=True, alpha=0.5, color='gray', label='Samples')

# Plot individual components and overall mixture
x_range = np.linspace(samples.min(), samples.max(), 1000)
mixture_pdf = np.zeros_like(x_range)

for i, (mean, std, weight) in enumerate(zip(means, stds, weights)):
    component_pdf = weight * norm.pdf(x_range, mean, std)
    mixture_pdf += component_pdf
    ax.plot(x_range, component_pdf, '--', alpha=0.7, linewidth=2, 
            label=f'Component {i+1}: μ={mean}, σ={std:.2f}, w={weight:.2f}')

ax.plot(x_range, mixture_pdf, 'r-', linewidth=3, label='True Mixture')

ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Mixture of Gaussians - Data Generation', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mixture_illustration.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved mixture illustration to 'mixture_illustration.png'")

# Hyperparameters for priors
K = 3  # Number of components

# Prior for π: Dirichlet(α)
alpha_prior = np.ones(K)  # Uniform prior

# Prior for μ: N(m0, s0^2)
m0 = samples.mean()
s0_squared = samples.var()

print(f"\nPrior hyperparameters:")
print(f"  Dirichlet α: {alpha_prior}")
print(f"  Normal m0: {m0:.2f}, s0²: {s0_squared:.2f}")

# Initialize variational parameters
# q(π) = Dirichlet(α')
alpha_variational = alpha_prior 

# q(μk) = N(mk', sk'^2)
m_var = np.random.normal(m0, np.sqrt(s0_squared), K)
s_squared_var = np.ones(K) * s0_squared / 2

# q(zi) = Categorical(φi)
phi = np.random.dirichlet(np.ones(K), n_samples)

print(f"\nInitial variational parameters:")
print(f"  α': {alpha_variational}")
print(f"  m': {m_var}")
print(f"  s'²: {s_squared_var}")


def compute_elbo(X, phi, alpha_variational, m_var, s_squared_var, 
                 alpha_prior, m0, s0_squared, sigma_squared):
    """
    Compute the Evidence Lower Bound (ELBO)
    
    ELBO = E_q[log p(x, z, π, μ)] - E_q[log q(z, π, μ)]
    """
    n = len(X)
    K = len(alpha_variational)
    
    elbo = 0.0
    
    # E_q[log p(x | z, μ)]
    # For each sample i and component k: φ_ik * log N(xi | mk', σ²)
    for i in range(n):
        for k in range(K):
            log_likelihood = norm.logpdf(X[i], loc=m_var[k], scale=np.sqrt(sigma_squared))
            elbo += phi[i, k] * log_likelihood
    
    # E_q[log p(z | π)]
    # For each sample i and component k: φ_ik * E[log πk]
    # E[log πk] under Dirichlet = ψ(α'k) - ψ(Σα'k)
    digamma_sum = digamma(alpha_variational.sum())
    for i in range(n):
        for k in range(K):
            expected_log_pi = digamma(alpha_variational[k]) - digamma_sum
            elbo += phi[i, k] * expected_log_pi
    
    # E_q[log p(π)] - Dirichlet prior
    # log p(π) = log Dirichlet(π | α)
    # E_q[log p(π)] = log Γ(Σα) - Σ log Γ(α) + Σ(α-1)E[log π]
    elbo += gammaln(alpha_prior.sum()) - gammaln(alpha_prior).sum()
    for k in range(K):
        expected_log_pi = digamma(alpha_variational[k]) - digamma_sum
        elbo += (alpha_prior[k] - 1) * expected_log_pi
    
    # E_q[log p(μ)] - Normal prior
    # For each component k: log N(μk | m0, s0²)
    for k in range(K):
        # E_q[log N(μk | m0, s0²)]
        # = -0.5*log(2π*s0²) - 0.5*E[(μk - m0)²]/s0²
        # E[(μk - m0)²] = (mk' - m0)² + sk'²
        expected_squared_diff = (m_var[k] - m0)**2 + s_squared_var[k]
        log_prior = -0.5 * np.log(2 * np.pi * s0_squared) - 0.5 * expected_squared_diff / s0_squared
        elbo += log_prior
    
    # -E_q[log q(z)]
    # Entropy of categorical: -Σi Σk φ_ik log φ_ik
    for i in range(n):
        for k in range(K):
            if phi[i, k] > 1e-10:  # Avoid log(0)
                elbo -= phi[i, k] * np.log(phi[i, k])
    
    # -E_q[log q(π)]
    # Entropy of Dirichlet
    elbo -= gammaln(alpha_variational.sum()) - gammaln(alpha_variational).sum()
    for k in range(K):
        expected_log_pi = digamma(alpha_variational[k]) - digamma_sum
        elbo -= (alpha_variational[k] - 1) * expected_log_pi
    
    # -E_q[log q(μ)]
    # Entropy of Gaussians: 0.5*log(2πe*s²)
    for k in range(K):
        entropy = 0.5 * np.log(2 * np.pi * np.e * s_squared_var[k])
        elbo += entropy
    
    return elbo


def update_phi(X, alpha_variational, m_var, s_squared_var, sigma_squared):
    """
    Update variational parameters for q(z)
    
    Mathematical Derivation:
    -----------------------
    We want to find the optimal q(z_i) that maximizes the ELBO. Taking the derivative
    of the ELBO with respect to q(z_i) and setting it to zero, we get:
    
    log q*(z_i = k) ∝ E_{q(π,μ)}[log p(x_i, z_i = k, π, μ)]
    
    Expanding the joint probability:
    log q*(z_i = k) ∝ E_q[log p(x_i | z_i = k, μ_k)] + E_q[log p(z_i = k | π)]
                    ∝ E_q[log N(x_i | μ_k, σ²)] + E_q[log π_k]
    
    For the likelihood term:
    E_q[log N(x_i | μ_k, σ²)] = E_q[-0.5 log(2πσ²) - 0.5(x_i - μ_k)²/σ²]
                               ≈ log N(x_i | m_k', σ²)  (using variational mean m_k')
    
    For the prior term:
    E_q[log π_k] = ψ(α'_k) - ψ(Σ_j α'_j)  (property of Dirichlet distribution)
    
    where ψ is the digamma function.
    
    Therefore:
    φ_ik = q(z_i = k) ∝ exp(E[log π_k] + E[log N(x_i | μ_k, σ²)])
    
    We normalize φ_ik so that Σ_k φ_ik = 1 for each sample i.
    """
    n = len(X)
    K = len(alpha_variational)
    phi = np.zeros((n, K))
    
    digamma_sum = digamma(alpha_variational.sum())
    
    for i in range(n):
        log_phi = np.zeros(K)
        for k in range(K):
            # E[log πk] = ψ(α'_k) - ψ(Σ α'_j)
            expected_log_pi = digamma(alpha_variational[k]) - digamma_sum
            
            # E[log N(xi | μk, σ²)] ≈ log N(xi | mk', σ²)
            log_likelihood = norm.logpdf(X[i], loc=m_var[k], scale=np.sqrt(sigma_squared))
            
            log_phi[k] = expected_log_pi + log_likelihood
        
        # Normalize (in log space for numerical stability)
        log_phi -= logsumexp(log_phi)
        phi[i] = np.exp(log_phi)
    
    return phi


def update_alpha(phi, alpha_prior):
    """
    Update variational parameters for q(π)
    α'k = αk + Σi φ_ik
    """
    return alpha_prior + phi.sum(axis=0)


def update_mu(X, phi, m0, s0_squared, sigma_squared):
    """
    Update variational parameters for q(μ)
    Using conjugate Gaussian updates
    """
    K = phi.shape[1]
    m_var = np.zeros(K)
    s_squared_var = np.zeros(K)
    
    for k in range(K):
        # Precision (inverse variance)
        precision_prior = 1.0 / s0_squared
        precision_likelihood = phi[:, k].sum() / sigma_squared
        precision_post = precision_prior + precision_likelihood
        
        s_squared_var[k] = 1.0 / precision_post
        
        # Mean
        mean_prior_term = m0 * precision_prior
        mean_likelihood_term = (phi[:, k] * X).sum() / sigma_squared
        m_var[k] = (mean_prior_term + mean_likelihood_term) / precision_post
    
    return m_var, s_squared_var


# Stochastic Variational Inference
n_iterations = 100
batch_size = 100
elbo_history = []

print(f"\nRunning Stochastic Variational Inference for {n_iterations} iterations...")
print(f"Batch size: {batch_size}")

for iteration in range(n_iterations):
    # Sample a mini-batch
    batch_indices = np.random.choice(n_samples, batch_size, replace=False)
    X_batch = samples[batch_indices]
    
    # Update phi for the batch
    phi_batch = update_phi(X_batch, alpha_variational, m_var, s_squared_var, sigma_squared)
    
    # Scale updates by n/batch_size for unbiased estimates
    scale = n_samples / batch_size
    
    # Update alpha (using scaled batch statistics)
    phi_scaled = phi_batch.sum(axis=0) * scale
    alpha_variational = alpha_prior + phi_scaled
    
    # Update mu (using batch)
    m_var, s_squared_var = update_mu(X_batch, phi_batch, m0, s0_squared, sigma_squared)
    
    # Update full phi for ELBO computation (every 10 iterations)
    if iteration % 10 == 0:
        phi = update_phi(samples, alpha_variational, m_var, s_squared_var, sigma_squared)
        elbo = compute_elbo(X=samples,phi=phi, alpha_variational=alpha_variational, m_var=m_var, s_squared_var=s_squared_var,
                           alpha_prior=alpha_prior, m0=m0, s0_squared=s0_squared, sigma_squared=sigma_squared)
        elbo_history.append(elbo)
        
        # Compute estimated weights
        estimated_weights = alpha_variational / alpha_variational.sum()
        
        print(f"\nIteration {iteration}:")
        print(f"  ELBO: {elbo:.2f}")
        print(f"  Estimated means: {m_var}")
        print(f"  Estimated weights: {estimated_weights}")

# Final update with full dataset
phi = update_phi(samples, alpha_variational, m_var, s_squared_var, sigma_squared)
alpha_variational = update_alpha(phi, alpha_prior)
m_var, s_squared_var = update_mu(samples, phi, m0, s0_squared, sigma_squared)

# Final results
estimated_weights = alpha_variational / alpha_variational.sum()
estimated_stds = np.sqrt(s_squared_var)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print("\nTrue parameters:")
print(f"  Means: {means}")
print(f"  Weights: {weights}")
print(f"  Std devs: {stds}")

print("\nEstimated parameters:")
print(f"  Means: {m_var}")
print(f"  Weights: {estimated_weights}")
print(f"  Std devs: {estimated_stds}")

# Sort by mean for comparison
true_order = np.argsort(means)
est_order = np.argsort(m_var)

print("\nSorted comparison:")
for i, (t_idx, e_idx) in enumerate(zip(true_order, est_order)):
    print(f"  Component {i+1}:")
    print(f"    True:      mean={means[t_idx]:.2f}, weight={weights[t_idx]:.3f}")
    print(f"    Estimated: mean={m_var[e_idx]:.2f}, weight={estimated_weights[e_idx]:.3f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: ELBO convergence
axes[0, 0].plot(range(0, n_iterations, 10), elbo_history, 'b-', linewidth=2)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('ELBO')
axes[0, 0].set_title('ELBO Convergence')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Data histogram with fitted components
axes[0, 1].hist(samples, bins=50, density=True, alpha=0.5, color='gray', label='Data')

# Plot true mixture
x_range = np.linspace(samples.min(), samples.max(), 1000)
true_mixture = sum(w * norm.pdf(x_range, m, s) for w, m, s in zip(weights, means, stds))
axes[0, 1].plot(x_range, true_mixture, 'r-', linewidth=2, label='True mixture')

# Plot estimated mixture
est_mixture = sum(w * norm.pdf(x_range, m, s) for w, m, s in zip(estimated_weights, m_var, estimated_stds))
axes[0, 1].plot(x_range, est_mixture, 'b--', linewidth=2, label='Estimated mixture')

axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Mixture Model Fit')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Component assignments (responsibility matrix)
# Show responsibilities for first 100 samples
phi_subset = phi[:100]
im = axes[1, 0].imshow(phi_subset.T, aspect='auto', cmap='viridis', interpolation='nearest')
axes[1, 0].set_xlabel('Sample index')
axes[1, 0].set_ylabel('Component')
axes[1, 0].set_title('Component Responsibilities (first 100 samples)')
axes[1, 0].set_yticks([0, 1, 2])
axes[1, 0].set_yticklabels(['Component 1', 'Component 2', 'Component 3'])
plt.colorbar(im, ax=axes[1, 0], label='Responsibility')

# Plot 4: Comparison of parameters
x_pos = np.arange(K)
width = 0.35

axes[1, 1].bar(x_pos - width/2, [means[i] for i in true_order], width, 
               label='True means', alpha=0.7)
axes[1, 1].bar(x_pos + width/2, [m_var[i] for i in est_order], width,
               label='Estimated means', alpha=0.7)
axes[1, 1].set_xlabel('Component (sorted by mean)')
axes[1, 1].set_ylabel('Mean value')
axes[1, 1].set_title('Comparison of Component Means')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(['Low', 'Medium', 'High'])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('svi_mixture_results.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'svi_mixture_results.png'")

plt.show()

print("\n" + "="*60)
print("SVI COMPLETE")
print("="*60)

