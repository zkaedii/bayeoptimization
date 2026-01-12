# 🔱 ZKAEDI PRIME

> **Recursively Coupled Hamiltonian Framework for ML/AI Optimization, Evidential Learning, and Uncertainty Quantification**

[![npm version](https://img.shields.io/npm/v/@zkaedi/zkaedi-prime)](https://www.npmjs.com/package/@zkaedi/zkaedi-prime)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3+-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)

A powerful TypeScript framework for advanced machine learning and AI optimization, featuring:

- **Bayesian Optimization** with Gaussian Process surrogates
- **Evidential Deep Learning** with uncertainty quantification
- **Active Learning** with intelligent query selection
- **Multimodal Fusion** via Hamiltonian consensus
- **Adversarial Robustness** with Hamiltonian smoothing
- **Open-Set Recognition** for novel class detection
- **Drift Detection** for temporal monitoring

---

## 🚀 Installation

### GitHub Packages

```bash
# Configure npm to use GitHub Packages
echo "@zkaedi:registry=https://npm.pkg.github.com" >> .npmrc
echo "//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}" >> .npmrc

# Install the package
npm install @zkaedi/zkaedi-prime
```

### NPM (if published)

```bash
npm install @zkaedi/zkaedi-prime
```

---

## 📦 Quick Start

### Bayesian Optimization

```typescript
import { BayesianOptimizer } from "@zkaedi/zkaedi-prime/optimization";

const optimizer = new BayesianOptimizer(
  {
    x: [-5, 5],
    y: [-5, 5],
  },
  { nIter: 50, nWarmup: 10 }
);

const result = await optimizer.optimize(async (params) => {
  const [x, y] = params;
  // Your expensive black-box function
  return -(x ** 2 + y ** 2); // Minimize negative (maximize)
});

console.log(`Best: ${result.bestX}, Value: ${result.bestY}`);
```

### Evidential Classification

```typescript
import { EvidentialClassifier } from "@zkaedi/zkaedi-prime/evidential";

const classifier = new EvidentialClassifier({
  nClasses: 10,
  klWeight: 0.001,
});

// Forward pass with evidence
const evidence = [2.5, 0.3, 0.1, 1.2, 0.2, 0.5, 0.8, 0.4, 0.6, 0.3];
const output = classifier.forward(evidence);
const prediction = classifier.predict(output);

console.log(`Predicted: ${prediction.predictedClass}`);
console.log(`Confidence: ${prediction.confidence}`);
console.log(`Uncertainty: ${prediction.uncertainty}`);
```

### Active Learning

```typescript
import { ZkaediPrimeActiveLearning } from "@zkaedi/zkaedi-prime/learning";

const activeLearner = new ZkaediPrimeActiveLearning({
  wUncertainty: 0.6,
  wDiversity: 0.1,
});

const selected = activeLearner.selectBatch(unlabeledData, model, budget=100);
console.log(`Selected ${selected.selectedIndices.length} samples`);
```

---

## 📚 Documentation

### Core Modules

#### **Optimization** (`@zkaedi/zkaedi-prime/optimization`)

- `BayesianOptimizer` - Gaussian Process-based optimization
- `ZkaediPrimeBO` - ZKAEDI PRIME enhanced Bayesian Optimization

**Example:**
```typescript
import { BayesianOptimizer } from "@zkaedi/zkaedi-prime/optimization";

const optimizer = new BayesianOptimizer(bounds, { nIter: 50 });
const result = await optimizer.optimize(objective);
```

#### **Evidential Learning** (`@zkaedi/zkaedi-prime/evidential`)

- `EvidentialClassifier` - Classification with uncertainty
- `EvidentialRegressor` - Regression with uncertainty
- `ZkaediPrimeOpenSetRecognition` - Open-set recognition
- `ZkaediPrimeMultimodalFusion` - Multimodal fusion

**Example:**
```typescript
import { EvidentialClassifier } from "@zkaedi/zkaedi-prime/evidential";

const classifier = new EvidentialClassifier({ nClasses: 10 });
const output = classifier.forward(evidence);
const prediction = classifier.predict(output);
```

#### **Security** (`@zkaedi/zkaedi-prime/security`)

- `ConfusionMatrixDefense` - FP/FN optimization
- `FalseNegativeHardening` - FN minimization strategies
- `ZkaediPrimeAdversarialRobustness` - Adversarial defense

**Example:**
```typescript
import { ConfusionMatrixDefense } from "@zkaedi/zkaedi-prime/security";

const defense = new ConfusionMatrixDefense({ fpBudget: 0.05, fnBudget: 0.01 });
defense.update(predicted, trueLabel, confidence);
const metrics = defense.computeMetrics();
```

#### **Learning** (`@zkaedi/zkaedi-prime/learning`)

- `ZkaediPrimeActiveLearning` - Active learning
- `ZkaediPrimeDriftDetector` - Temporal drift detection

**Example:**
```typescript
import { ZkaediPrimeActiveLearning } from "@zkaedi/zkaedi-prime/learning";

const activeLearner = new ZkaediPrimeActiveLearning();
const selected = activeLearner.selectBatch(data, model, budget);
```

---

## 🎯 Features

### ✅ Bayesian Optimization

- Gaussian Process surrogate models
- Multiple acquisition functions (EI, PI, UCB)
- RBF and Matern kernels
- Recursive Hamiltonian dynamics

### ✅ Evidential Learning

- Dirichlet-based classification
- Normal-Inverse-Gamma regression
- Epistemic uncertainty quantification
- Open-set recognition

### ✅ Active Learning

- Uncertainty-driven selection
- Diversity-based sampling
- Batch-mode acquisition
- Inter-sample repulsion

### ✅ Multimodal Fusion

- Dempster-Shafer combination
- Product-of-Experts
- Hamiltonian consensus
- Conflict detection

### ✅ Adversarial Robustness

- Hamiltonian Adversarial Training (HAT)
- Recursive smoothing
- Evidential uncertainty injection

### ✅ Drift Detection

- KL divergence monitoring
- Maximum Mean Discrepancy (MMD)
- Adaptive retraining triggers

---

## 📖 Examples

See the [examples directory](./examples) for complete usage examples:

- `bayesianOptimization.ts` - Optimization examples
- `evidentialLearning.ts` - Evidential learning examples

---

## 🔧 Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Type check
npm run type-check

# Test
npm test

# Lint
npm run lint

# Format
npm run format
```

---

## 📝 License

MIT License - see [LICENSE](./LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please read our [Contributing Guide](./CONTRIBUTING.md) first.

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/zkaedi/zkaedi-prime/issues)
- **Discussions**: [GitHub Discussions](https://github.com/zkaedi/zkaedi-prime/discussions)

---

## 🙏 Acknowledgments

Built with the ZKAEDI PRIME framework — recursively coupled Hamiltonian dynamics for ML/AI.

---

**Made with 🔱 by ZKAEDI**
