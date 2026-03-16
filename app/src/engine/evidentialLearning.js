/**
 * ZKAEDI PRIME - Evidential Learning Engine
 * Dirichlet-based classification + Normal-Inverse-Gamma regression
 * with uncertainty quantification
 */

class EvidentialClassifier {
  constructor(nClasses = 5) {
    this.nClasses = nClasses;
    this.history = [];
  }

  // Convert raw evidence to Dirichlet parameters
  forward(evidence) {
    // evidence: array of non-negative values per class
    const alpha = evidence.map(e => e + 1); // Dirichlet concentration
    const S = alpha.reduce((a, b) => a + b, 0); // Dirichlet strength
    const probabilities = alpha.map(a => a / S);
    const uncertainty = this.nClasses / S; // Vacuity uncertainty

    // Dissonance (conflicting evidence)
    let dissonance = 0;
    for (let i = 0; i < this.nClasses; i++) {
      let bal = 0;
      for (let j = 0; j < this.nClasses; j++) {
        if (i !== j) {
          const denom = alpha[i] + alpha[j];
          bal += (alpha[j] / denom) * (1 - Math.abs(alpha[i] - alpha[j]) / denom);
        }
      }
      dissonance += probabilities[i] * bal;
    }

    return {
      alpha,
      strength: S,
      probabilities,
      uncertainty,
      dissonance,
      evidence
    };
  }

  predict(evidence) {
    const output = this.forward(evidence);
    const predictedClass = output.probabilities.indexOf(Math.max(...output.probabilities));
    const confidence = output.probabilities[predictedClass];

    const result = {
      predictedClass,
      confidence,
      uncertainty: output.uncertainty,
      dissonance: output.dissonance,
      probabilities: output.probabilities,
      alpha: output.alpha,
      isHighUncertainty: output.uncertainty > 0.5
    };

    this.history.push(result);
    return result;
  }

  // Generate synthetic data for demo
  generateSyntheticBatch(batchSize = 20) {
    const results = [];
    for (let i = 0; i < batchSize; i++) {
      // Generate evidence with varying certainty
      const certaintyLevel = Math.random();
      const trueClass = Math.floor(Math.random() * this.nClasses);
      const evidence = Array.from({ length: this.nClasses }, (_, c) => {
        if (c === trueClass) {
          return certaintyLevel * 10 + Math.random() * 2;
        }
        return Math.random() * (1 - certaintyLevel) * 3;
      });

      results.push({
        evidence,
        trueClass,
        prediction: this.predict(evidence)
      });
    }
    return results;
  }
}

class EvidentialRegressor {
  constructor() {
    this.history = [];
  }

  // Normal-Inverse-Gamma prediction
  predict(gamma, v, alpha, beta) {
    // gamma: predicted mean
    // v: virtual observations
    // alpha: shape parameter
    // beta: scale parameter
    const mean = gamma;
    const aleatoricUncertainty = beta / (alpha - 1); // Expected data noise
    const epistemicUncertainty = beta / (v * (alpha - 1)); // Model uncertainty
    const totalUncertainty = aleatoricUncertainty + epistemicUncertainty;

    const result = {
      mean,
      aleatoricUncertainty,
      epistemicUncertainty,
      totalUncertainty,
      confidence: 1 / (1 + totalUncertainty),
      lowerBound: mean - 2 * Math.sqrt(totalUncertainty),
      upperBound: mean + 2 * Math.sqrt(totalUncertainty)
    };

    this.history.push(result);
    return result;
  }

  // Generate synthetic regression data
  generateSyntheticSeries(nPoints = 50) {
    const results = [];
    for (let i = 0; i < nPoints; i++) {
      const x = (i / nPoints) * 10;
      const trueY = Math.sin(x) * 2 + Math.cos(x * 0.5);

      // Simulate NIG parameters with varying certainty
      const distFromTraining = Math.min(Math.abs(x - 3), Math.abs(x - 7)) / 5;
      const gamma = trueY + (Math.random() - 0.5) * 0.3;
      const v = Math.max(2, 10 - distFromTraining * 8);
      const alpha = Math.max(1.5, 5 - distFromTraining * 3);
      const beta = 0.5 + distFromTraining * 2;

      results.push({
        x,
        trueY,
        prediction: this.predict(gamma, v, alpha, beta)
      });
    }
    return results;
  }
}

class OpenSetRecognizer {
  constructor(nKnownClasses = 5, rejectionThreshold = 0.6) {
    this.nKnownClasses = nKnownClasses;
    this.rejectionThreshold = rejectionThreshold;
    this.classifier = new EvidentialClassifier(nKnownClasses);
  }

  recognize(evidence) {
    const prediction = this.classifier.predict(evidence);
    const isUnknown = prediction.uncertainty > this.rejectionThreshold;

    return {
      ...prediction,
      isUnknown,
      label: isUnknown ? 'UNKNOWN' : `Class ${prediction.predictedClass}`,
      rejectionThreshold: this.rejectionThreshold
    };
  }

  // Generate mixed known/unknown samples
  generateMixedBatch(nKnown = 15, nUnknown = 5) {
    const results = [];

    // Known class samples
    for (let i = 0; i < nKnown; i++) {
      const trueClass = Math.floor(Math.random() * this.nKnownClasses);
      const evidence = Array.from({ length: this.nKnownClasses }, (_, c) => {
        if (c === trueClass) return 5 + Math.random() * 10;
        return Math.random() * 1.5;
      });
      results.push({ type: 'known', trueClass, result: this.recognize(evidence) });
    }

    // Unknown class samples (low evidence across all classes)
    for (let i = 0; i < nUnknown; i++) {
      const evidence = Array.from({ length: this.nKnownClasses }, () =>
        Math.random() * 0.8
      );
      results.push({ type: 'unknown', trueClass: -1, result: this.recognize(evidence) });
    }

    return results.sort(() => Math.random() - 0.5);
  }
}

module.exports = { EvidentialClassifier, EvidentialRegressor, OpenSetRecognizer };
