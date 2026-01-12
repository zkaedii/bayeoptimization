# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added

- **Bayesian Optimization**
  - Gaussian Process surrogate models
  - Multiple acquisition functions (EI, PI, UCB)
  - RBF and Matern kernels
  - ZKAEDI PRIME recursive Hamiltonian dynamics

- **Evidential Deep Learning**
  - Dirichlet-based classification with uncertainty quantification
  - Normal-Inverse-Gamma regression
  - Open-set recognition
  - Multimodal fusion via Hamiltonian consensus

- **Active Learning**
  - Uncertainty-driven query selection
  - Diversity-based sampling
  - Batch-mode acquisition with inter-sample repulsion

- **Security & Robustness**
  - Confusion matrix defense for FP/FN optimization
  - False negative hardening strategies
  - Adversarial robustness with Hamiltonian smoothing

- **Temporal Monitoring**
  - Drift detection via KL divergence and MMD
  - Adaptive retraining triggers
  - Style consistency monitoring

### Documentation

- Complete README with examples
- API documentation
- Usage guides for all modules

### Infrastructure

- TypeScript support with full type definitions
- ESM and CommonJS builds
- Comprehensive test suite
- GitHub Packages publishing configuration

---

## [Unreleased]

### Planned

- Additional kernel functions for GP
- More acquisition function variants
- Extended evidential learning models
- Performance optimizations
