/**
 * ZKAEDI PRIME - Adversarial Robustness Lab
 * FGSM attacks, Hamiltonian smoothing, robustness evaluation
 */

class AdversarialLab {
  constructor(options = {}) {
    this.epsilon = options.epsilon || 0.1;
    this.nClasses = options.nClasses || 5;
    this.smoothingStrength = options.smoothingStrength || 0.3;
  }

  // Simulate a simple neural network decision boundary
  _simulateModel(input) {
    // Create a multi-class decision from input features
    const logits = Array.from({ length: this.nClasses }, (_, c) => {
      let score = 0;
      for (let i = 0; i < input.length; i++) {
        // Different weight patterns per class
        score += input[i] * Math.sin((c + 1) * (i + 1) * 0.7) * 2;
      }
      return score;
    });

    // Softmax
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(e => e / sumExp);

    return { logits, probabilities: probs };
  }

  // FGSM attack: generate adversarial perturbation
  fgsm(input, targetClass = null) {
    const original = this._simulateModel(input);
    const originalClass = original.probabilities.indexOf(Math.max(...original.probabilities));

    // Numerical gradient of loss w.r.t. input
    const gradients = input.map((val, i) => {
      const eps = 0.001;
      const inputPlus = [...input];
      inputPlus[i] += eps;
      const outputPlus = this._simulateModel(inputPlus);

      // Cross-entropy loss gradient approximation
      const target = targetClass !== null ? targetClass : originalClass;
      return -(outputPlus.probabilities[target] - original.probabilities[target]) / eps;
    });

    // Sign of gradient * epsilon
    const perturbation = gradients.map(g => this.epsilon * Math.sign(g));
    const adversarial = input.map((val, i) => val + perturbation[i]);

    const adversarialOutput = this._simulateModel(adversarial);
    const adversarialClass = adversarialOutput.probabilities.indexOf(
      Math.max(...adversarialOutput.probabilities)
    );

    return {
      originalInput: input,
      adversarialInput: adversarial,
      perturbation,
      perturbationNorm: Math.sqrt(perturbation.reduce((a, b) => a + b * b, 0)),
      originalClass,
      adversarialClass,
      attackSuccess: originalClass !== adversarialClass,
      originalConfidence: original.probabilities[originalClass],
      adversarialConfidence: adversarialOutput.probabilities[adversarialClass],
      originalProbs: original.probabilities,
      adversarialProbs: adversarialOutput.probabilities
    };
  }

  // Hamiltonian smoothing defense
  hamiltonianSmoothing(input) {
    const nSamples = 20;
    const smoothed = new Float64Array(input.length);
    const allPredictions = [];

    for (let s = 0; s < nSamples; s++) {
      // Add controlled Hamiltonian noise
      const noisy = input.map(val =>
        val + (Math.random() - 0.5) * this.smoothingStrength * 2
      );
      const pred = this._simulateModel(noisy);
      allPredictions.push(pred.probabilities);
      for (let i = 0; i < input.length; i++) {
        smoothed[i] += noisy[i] / nSamples;
      }
    }

    // Average predictions
    const avgProbs = Array.from({ length: this.nClasses }, (_, c) => {
      return allPredictions.reduce((sum, p) => sum + p[c], 0) / nSamples;
    });

    // Prediction variance (robustness indicator)
    const predVariance = Array.from({ length: this.nClasses }, (_, c) => {
      const mean = avgProbs[c];
      return allPredictions.reduce((sum, p) => sum + (p[c] - mean) ** 2, 0) / nSamples;
    });

    const predictedClass = avgProbs.indexOf(Math.max(...avgProbs));

    return {
      smoothedInput: Array.from(smoothed),
      probabilities: avgProbs,
      predictedClass,
      confidence: avgProbs[predictedClass],
      predictionVariance: predVariance,
      robustnessScore: 1 - Math.max(...predVariance) * 10
    };
  }

  // Run full robustness evaluation
  evaluateRobustness(nSamples = 30) {
    const results = {
      attacks: [],
      defenses: [],
      summary: {}
    };

    let attackSuccesses = 0;
    let defenseSuccesses = 0;

    for (let i = 0; i < nSamples; i++) {
      const inputDim = 8;
      const input = Array.from({ length: inputDim }, () => (Math.random() - 0.5) * 4);

      // Attack
      const attack = this.fgsm(input);
      results.attacks.push(attack);
      if (attack.attackSuccess) attackSuccesses++;

      // Defense
      const defense = this.hamiltonianSmoothing(attack.adversarialInput);
      const originalOutput = this._simulateModel(input);
      const originalClass = originalOutput.probabilities.indexOf(
        Math.max(...originalOutput.probabilities)
      );
      const defenseRestored = defense.predictedClass === originalClass;
      if (defenseRestored) defenseSuccesses++;

      results.defenses.push({
        ...defense,
        originalClass,
        restored: defenseRestored
      });
    }

    results.summary = {
      totalSamples: nSamples,
      attackSuccessRate: attackSuccesses / nSamples,
      defenseSuccessRate: defenseSuccesses / nSamples,
      averageRobustness: results.defenses.reduce((s, d) => s + d.robustnessScore, 0) / nSamples,
      averagePerturbationNorm: results.attacks.reduce((s, a) => s + a.perturbationNorm, 0) / nSamples
    };

    return results;
  }
}

module.exports = { AdversarialLab };
