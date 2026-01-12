/**
 * 🔱 ZKAEDI PRIME — Temporal Drift Detection
 * 
 * Monitor distribution shifts and trigger retraining
 */

export interface DriftDetectorOptions {
  windowSize?: number;
  threshold?: number;
}

export interface DriftDetectionResult {
  driftDetected: boolean;
  driftScore: number;
  phase: "STABLE" | "DRIFT" | "CRITICAL";
}

/**
 * Temporal Drift Detection via Hamiltonian Monitoring
 */
export class ZkaediPrimeDriftDetector {
  private windowSize: number;
  private threshold: number;
  private referenceData: Array<{ x: number[]; y: number }> = [];
  private window: Array<{ x: number[]; y: number }> = [];

  constructor(options: DriftDetectorOptions = {}) {
    this.windowSize = options.windowSize ?? 100;
    this.threshold = options.threshold ?? 0.1;
  }

  /**
   * Initialize reference distribution
   */
  initializeReference(data: Array<{ x: number[]; y: number }>): void {
    this.referenceData = [...data];
  }

  /**
   * Detect drift in new sample
   */
  detectDrift(sample: { x: number[]; y: number }): DriftDetectionResult {
    this.window.push(sample);
    if (this.window.length > this.windowSize) {
      this.window.shift();
    }

    if (this.referenceData.length === 0) {
      return {
        driftDetected: false,
        driftScore: 0,
        phase: "STABLE",
      };
    }

    // Compute KL divergence (simplified)
    const driftScore = this.computeDriftScore();

    const driftDetected = driftScore > this.threshold;
    const phase: "STABLE" | "DRIFT" | "CRITICAL" =
      driftScore > this.threshold * 2
        ? "CRITICAL"
        : driftDetected
          ? "DRIFT"
          : "STABLE";

    return {
      driftDetected,
      driftScore,
      phase,
    };
  }

  /**
   * Check if retraining is needed
   */
  shouldRetrain(): boolean {
    if (this.window.length < this.windowSize) return false;

    const driftScore = this.computeDriftScore();
    return driftScore > this.threshold * 1.5;
  }

  /**
   * Update reference with new data
   */
  updateReference(): void {
    this.referenceData = [...this.window];
  }

  private computeDriftScore(): number {
    // Simplified KL divergence approximation
    // In production, would use actual KL divergence or MMD
    if (this.referenceData.length === 0 || this.window.length === 0) return 0;

    // Compute mean difference
    const refMean = this.computeMean(this.referenceData);
    const winMean = this.computeMean(this.window);
    const diff = this.euclideanDistance(refMean, winMean);

    return diff;
  }

  private computeMean(data: Array<{ x: number[]; y: number }>): number[] {
    if (data.length === 0) return [];
    const dim = data[0].x.length;
    const mean = Array(dim).fill(0);
    for (const item of data) {
      for (let i = 0; i < dim; i++) {
        mean[i] += item.x[i];
      }
    }
    return mean.map((m) => m / data.length);
  }

  private euclideanDistance(a: number[], b: number[]): number {
    if (a.length !== b.length) return Infinity;
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }
}
