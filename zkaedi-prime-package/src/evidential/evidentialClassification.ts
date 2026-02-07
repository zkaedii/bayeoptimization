/**
 * 🔱 ZKAEDI PRIME — Evidential Classification
 * 
 * Evidential Deep Learning for classification with uncertainty quantification
 */

export interface EvidentialClassifierOptions {
  nClasses: number;
  klWeight?: number;
  klAnnealing?: boolean;
}

export interface EvidentialOutput {
  alpha: number[];
  uncertainty: number;
  evidence: number[];
}

export interface EvidentialPrediction {
  predictedClass: number;
  confidence: number;
  uncertainty: number;
  probabilities: number[];
}

/**
 * Evidential Classifier using Dirichlet distributions
 */
export class EvidentialClassifier {
  private readonly options: Required<EvidentialClassifierOptions>;

  constructor(options: EvidentialClassifierOptions) {
    this.options = {
      nClasses: options.nClasses,
      klWeight: options.klWeight ?? 0.001,
      klAnnealing: options.klAnnealing ?? true,
    };
  }

  /**
   * Get configuration options
   */
  getOptions(): Required<EvidentialClassifierOptions> {
    return { ...this.options };
  }

  /**
   * Forward pass: convert evidence to Dirichlet parameters
   */
  forward(evidence: number[]): EvidentialOutput {
    if (evidence.length !== this.options.nClasses) {
      throw new Error(`Evidence length ${evidence.length} != nClasses ${this.options.nClasses}`);
    }

    // Ensure non-negative evidence
    const safeEvidence = evidence.map((e) => Math.max(0, e));

    // Dirichlet parameters: alpha = evidence + 1
    const alpha = safeEvidence.map((e) => e + 1);
    const S = alpha.reduce((sum, a) => sum + a, 0);

    // Uncertainty: u = K / S
    const uncertainty = this.options.nClasses / S;

    return {
      alpha,
      uncertainty,
      evidence: safeEvidence,
    };
  }

  /**
   * Predict class from evidential output
   */
  predict(output: EvidentialOutput): EvidentialPrediction {
    const { alpha } = output;
    const S = alpha.reduce((sum, a) => sum + a, 0);

    // Expected probabilities: p_k = alpha_k / S
    const probabilities = alpha.map((a) => a / S);

    // Predicted class: argmax(p_k)
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));

    // Confidence: 1 - uncertainty
    const confidence = 1 - output.uncertainty;

    return {
      predictedClass,
      confidence,
      uncertainty: output.uncertainty,
      probabilities,
    };
  }

  /**
   * Check if uncertainty is high
   */
  isHighUncertainty(output: EvidentialOutput, threshold: number = 0.5): boolean {
    return output.uncertainty > threshold;
  }
}
