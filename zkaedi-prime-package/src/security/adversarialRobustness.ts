/**
 * 🔱 ZKAEDI PRIME — Adversarial Robustness
 * 
 * Hamiltonian Adversarial Training (HAT) for robustness
 */

export interface AdversarialRobustnessOptions {
  epsilon?: number;
  nSteps?: number;
  stepSize?: number;
}

export interface AdversarialRobustnessResult {
  cleanAccuracy: number;
  adversarialAccuracy: number;
  robustnessGain: number;
}

/**
 * Adversarial Robustness via Hamiltonian Smoothing
 */
export class ZkaediPrimeAdversarialRobustness {
  private epsilon: number;
  private _nSteps: number;
  private _stepSize: number;

  constructor(options: AdversarialRobustnessOptions = {}) {
    this.epsilon = options.epsilon ?? 0.1;
    this._nSteps = options.nSteps ?? 10;
    this._stepSize = options.stepSize ?? 0.01;
  }

  /**
   * Get number of steps
   */
  getNSteps(): number {
    return this._nSteps;
  }

  /**
   * Get step size
   */
  getStepSize(): number {
    return this._stepSize;
  }

  /**
   * Generate adversarial example (FGSM)
   */
  generateAdversarial(
    input: number[],
    gradient: number[],
    epsilon: number = this.epsilon
  ): number[] {
    return input.map((x, i) => {
      const perturbation = epsilon * Math.sign(gradient[i] || 0);
      return x + perturbation;
    });
  }

  /**
   * Evaluate robustness
   */
  evaluateRobustness(
    cleanPredictions: number[],
    adversarialPredictions: number[],
    trueLabels: number[]
  ): AdversarialRobustnessResult {
    const cleanCorrect = cleanPredictions.filter((p, i) => p === trueLabels[i]).length;
    const advCorrect = adversarialPredictions.filter((p, i) => p === trueLabels[i]).length;

    const cleanAccuracy = cleanCorrect / cleanPredictions.length;
    const adversarialAccuracy = advCorrect / adversarialPredictions.length;
    const robustnessGain = adversarialAccuracy - cleanAccuracy;

    return {
      cleanAccuracy,
      adversarialAccuracy,
      robustnessGain,
    };
  }
}
