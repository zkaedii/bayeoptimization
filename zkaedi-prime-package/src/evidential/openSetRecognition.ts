/**
 * 🔱 ZKAEDI PRIME — Open-Set Recognition
 * 
 * Detect and reject novel/unknown classes
 */

export interface OpenSetRecognitionOptions {
  numKnownClasses: number;
  rejectionThreshold?: number;
}

export interface OpenSetResult {
  predictedClass: number;
  isUnknown: boolean;
  confidence: number;
  uncertainty: number;
}

/**
 * Open-Set Recognition using evidential uncertainty
 */
export class ZkaediPrimeOpenSetRecognition {
  private numKnownClasses: number;
  private rejectionThreshold: number;

  constructor(options: OpenSetRecognitionOptions) {
    this.numKnownClasses = options.numKnownClasses;
    this.rejectionThreshold = options.rejectionThreshold ?? 0.5;
  }

  /**
   * Classify with open-set detection
   */
  classify(evidence: number[]): OpenSetResult {
    if (evidence.length < this.numKnownClasses) {
      throw new Error(`Evidence length ${evidence.length} < numKnownClasses ${this.numKnownClasses}`);
    }

    // Compute total evidence
    const totalEvidence = evidence.slice(0, this.numKnownClasses).reduce((sum, e) => sum + e, 0);
    const S = totalEvidence + this.numKnownClasses;

    // Uncertainty: u = K / S
    const uncertainty = this.numKnownClasses / S;

    // Check if unknown
    const isUnknown = uncertainty > this.rejectionThreshold;

    // Predict class from known classes
    const knownEvidence = evidence.slice(0, this.numKnownClasses);
    const probabilities = knownEvidence.map((e) => (e + 1) / S);
    const predictedClass = probabilities.indexOf(Math.max(...probabilities));

    // Confidence: 1 - uncertainty
    const confidence = 1 - uncertainty;

    return {
      predictedClass,
      isUnknown,
      confidence,
      uncertainty,
    };
  }
}
