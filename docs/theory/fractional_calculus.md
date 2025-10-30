# Fractional Calculus: Theory and Applications

## Table of Contents
1. [Introduction](#introduction)
2. [Definitions](#definitions)
3. [Properties](#properties)
4. [Numerical Methods](#numerical-methods)
5. [Applications in Neuroimaging](#applications)

## Introduction

Fractional calculus extends differentiation and integration to non-integer orders. Unlike integer-order derivatives that are local operators, **fractional derivatives are non-local**, capturing memory effects and long-range dependencies.

### Motivation for Neuroimaging

Biological systems exhibit:
- **Memory effects**: Current state depends on entire history
- **Non-Markovian dynamics**: Violations of memorylessness
- **Anomalous diffusion**: Non-Gaussian, sub/superdiffusive transport
- **Power-law relaxation**: Stretched exponentials, long tails

Traditional integer-order models cannot capture these phenomena. Fractional calculus provides the mathematical framework.

## Definitions

### 1. Riemann-Liouville Fractional Derivative

The classical definition via fractional integral:

```
D^α_t f(t) = 1/Γ(n-α) · d^n/dt^n ∫₀ᵗ (t-τ)^{n-α-1} f(τ) dτ
```

where n = ⌈α⌉ (ceiling of α).

**For α ∈ (0,1)**:
```
D^α_t f(t) = 1/Γ(1-α) · d/dt ∫₀ᵗ f(τ)/(t-τ)^α dτ
```

**Key property**: D^α(1) ≠ 0 (derivative of constant is non-zero!)

### 2. Caputo Fractional Derivative

The preferred definition for physical applications:

```
ᶜD^α_t f(t) = 1/Γ(n-α) ∫₀ᵗ f^{(n)}(τ)/(t-τ)^{α-n+1} dτ
```

**For α ∈ (0,1)**:
```
ᶜD^α_t f(t) = 1/Γ(1-α) ∫₀ᵗ f'(τ)/(t-τ)^α dτ
```

**Key property**: ᶜD^α(c) = 0 for constant c (matches physical intuition)

**Advantages for physics**:
- Initial conditions: y(0), y'(0), ... have clear meaning
- Derivative of constant is zero
- Better for solving fractional ODEs/PDEs

### 3. Grünwald-Letnikov Derivative

Direct discretization (useful for numerical computation):

```
D^α f(t) = lim_{h→0} h^{-α} Σₖ₌₀^{⌊t/h⌋} w_k^{(α)} f(t - kh)
```

where weights are computed recursively:
```
w_0^{(α)} = 1
w_k^{(α)} = (1 - (α+1)/k) · w_{k-1}^{(α)}
```

**Converges to Riemann-Liouville as h → 0**

### 4. Fractional Laplacian

Spatial fractional derivative via Fourier transform:

```
F[(-Δ)^{s} u](ξ) = |ξ|^{2s} F[u](ξ)
```

**Integral formulation** (for s ∈ (0,1)):
```
(-Δ)^{s} u(x) = C(n,s) P.V. ∫_{R^n} [u(x) - u(y)]/|x-y|^{n+2s} dy
```

where C(n,s) = 2^{2s} Γ((n+2s)/2) / (π^{n/2} Γ(-s))

**Applications**:
- Anomalous diffusion in tissue
- Lévy flights
- Non-local interactions

## Properties

### Linearity
```
D^α[a·f(t) + b·g(t)] = a·D^α f(t) + b·D^α g(t)
```

### Composition (Caution!)
```
D^α D^β f ≠ D^{α+β} f  (in general)
```

Composition rules depend on initial conditions and definition used.

### Power Functions
Analytical result for t^p:

**Caputo**:
```
ᶜD^α(t^p) = Γ(p+1)/Γ(p-α+1) · t^{p-α}  (p > α)
            0                             (p ≤ α, p ∈ N)
```

**Example**: ᶜD^{0.5}(t²) = 2·Γ(3)/Γ(2.5) · t^{1.5} = 4t^{1.5}/√π

### Exponential Functions
No simple closed form in general. For special cases:

```
D^α e^{λt} ≈ λ^α e^{λt}  (approximation for small α)
```

### Leibniz Rule (Product Rule)
Complicated! Involves infinite series:

```
D^α[f(t)g(t)] = Σₖ₌₀^∞ (α choose k) D^k f(t) · D^{α-k} g(t)
```

where (α choose k) = α(α-1)...(α-k+1)/k!

## Numerical Methods

### L1 Scheme (First-Order)

Discretize Caputo derivative at t_n:

```
D^α f(t_n) ≈ dt^{-α}/Γ(2-α) Σₖ₌₁ⁿ b_k [f(t_{n-k+1}) - f(t_{n-k})]
```

where b_k = k^{1-α} - (k-1)^{1-α}

**Convergence**: O(dt) for 0 < α < 1

**Stability**: Stable for appropriate dt (depends on α)

### L2 Scheme (Second-Order)

Uses three-point stencil:

```
D^α f(t_n) ≈ dt^{-α}/Γ(3-α) Σₖ₌₁ⁿ a_k [...]
```

**Convergence**: O(dt²)

### FFT-Based (Fractional Laplacian)

For periodic domains:

```
(-Δ)^{s} u = F^{-1}[|ξ|^{2s} F[u]]
```

**Computational cost**: O(N log N) via FFT

**Highly efficient for regular grids**

## Applications in Neuroimaging

### 1. Fractional BOLD Signal Modeling

**Classical balloon model**:
```
ds/dt = ε u(t) - κ s - γ(f - 1)
df/dt = s
```

**Fractional extension**:
```
D^α s(t) = ε u(t) - κ s - γ(f - 1)
D^β f(t) = s
```

**Physical interpretation**:
- α < 1: Memory in vasodilatory signal (smoothed response)
- β < 1: Delayed flow response (sluggish adjustment)

**Benefits**:
- Better fit to empirical hemodynamic response functions
- α as biomarker for neurovascular coupling
- Regional variation (sensory vs. default mode network)

### 2. Anomalous Diffusion in White Matter

**Classical diffusion**:
```
∂C/∂t = D ∇²C
```
Mean squared displacement: ⟨r²⟩ ∝ t

**Fractional diffusion**:
```
∂C/∂t = D_α (-Δ)^{α/2} C
```
Mean squared displacement: ⟨r²⟩ ∝ t^α

- α < 1: **Subdiffusion** (restricted diffusion, trapping)
- α = 1: Normal diffusion
- α > 1: **Superdiffusion** (ballistic transport)

**Applications**:
- Complex tissue microstructure
- Crossing fibers
- Partial volume effects

**Advantage over DTI/NODDI**: Captures non-Gaussian effects without multi-compartment assumptions

### 3. Functional Connectivity with Memory

**Classical Pearson correlation**: Assumes memoryless processes

**Fractional approach**:
```
Cov[X(t), Y(t+τ)] ∝ τ^{2H-2}
```

where H is Hurst exponent (H = 0.5 + d, d = fractional integration order)

**Reveals**:
- Long-range temporal correlations
- Persistent vs. anti-persistent dynamics
- Memory timescales across regions

### 4. Continuous-Time Random Walks (CTRW)

**Model for restricted diffusion**:

Particle performs random walk with:
- **Waiting times**: ψ(t) ∝ t^{-1-α}  (power-law)
- **Jump lengths**: p(r) ∝ r^{-1-β}   (heavy-tailed)

**Leads to fractional diffusion equation**:
```
∂^α C/∂t^α = K_β ∇^β C
```

**Captures**:
- Trapping in pores
- Tortuous paths
- Anomalous scaling

## Special Functions

### Mittag-Leffler Function

**Definition**:
```
E_{α,β}(z) = Σₖ₌₀^∞ z^k/Γ(αk + β)
```

**Special cases**:
- E_{1,1}(z) = e^z (exponential)
- E_{2,1}(-z²) = cos(z)

**Importance**: Fundamental solution of fractional ODEs

**Example**: Solution to D^α y = λy, y(0) = y₀
```
y(t) = y₀ E_{α,1}(λ t^α)
```

### Fractional Relaxation

**Equation**: D^α y(t) = -y(t)/τ^α

**Solution**: y(t) = y₀ E_{α,1}(-(t/τ)^α)

**Behavior**:
- α = 1: Exponential decay
- α < 1: Stretched exponential (slower decay)
- Power-law at long times: y(t) ∝ t^{-α}

## Convergence and Stability

### Convergence Rates

**Caputo L1 scheme**: O(Δt^{1-α} · Δt) = O(Δt^{2-α}) for smooth f

**Caputo L2 scheme**: O(Δt^{2})

**Grünwald-Letnikov**: O(Δt)

### Stability Conditions

**Fractional diffusion (explicit scheme)**:
```
Δt < C · (Δx)^α / D_α
```

where C depends on α (more restrictive for smaller α)

**CFL-like condition** but with fractional power of Δx

## References

1. **Podlubny, I. (1999)**. *Fractional Differential Equations*. Academic Press.
   - Comprehensive reference on fractional calculus

2. **Diethelm, K. (2010)**. *The Analysis of Fractional Differential Equations*. Springer.
   - Numerical methods and error analysis

3. **Magin, R. L. (2006)**. *Fractional Calculus in Bioengineering*. Begell House.
   - Applications to biological systems

4. **Metzler, R., & Klafter, J. (2000)**. The random walk's guide to anomalous diffusion. *Physics Reports*, 339, 1-77.
   - Continuous-time random walks and fractional diffusion

5. **West, B. J., Bologna, M., & Grigolini, P. (2003)**. *Physics of Fractal Operators*. Springer.
   - Physical interpretation and applications

## Summary

**Key takeaways**:
1. Fractional derivatives are **non-local** (memory effects)
2. **Caputo** preferred for physics (clear initial conditions)
3. **Mittag-Leffler** function is fractional analog of exponential
4. **Anomalous diffusion**: ⟨r²⟩ ∝ t^α captures non-Gaussian transport
5. Applications: BOLD dynamics, tissue microstructure, functional connectivity

**When to use fractional calculus**:
- Empirical data shows **power-law behavior** (long tails)
- **Memory effects** (past influences present)
- **Anomalous scaling** (non-Gaussian statistics)
- Standard models fail to fit biological data
