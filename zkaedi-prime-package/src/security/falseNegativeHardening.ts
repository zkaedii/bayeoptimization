/**
 * 🔱 ZKAEDI PRIME — False Negative Hardening
 * 
 * Minimize missed detections in high-stakes systems
 */

export interface FNHardeningOptions {
  costPositive?: number;
  costNegative?: number;
}

export interface FNHardeningResult {
  threshold: number;
  fnRate: number;
  fpRate: number;
  recall: number;
  precision: number;
}

/**
 * False Negative Hardening strategies
 */
export class FalseNegativeHardening {
  private costPositive: number;
  private costNegative: number;

  constructor(options: FNHardeningOptions = {}) {
    this.costPositive = options.costPositive ?? 1.0;
    this.costNegative = options.costNegative ?? 1.0;
  }

  /**
   * Optimize threshold to minimize FN
   */
  optimizeThreshold(
    scores: number[],
    labels: number[],
    thresholds: number[] = []
  ): FNHardeningResult {
    if (thresholds.length === 0) {
      // Generate thresholds
      const min = Math.min(...scores);
      const max = Math.max(...scores);
      for (let t = min; t <= max; t += (max - min) / 100) {
        thresholds.push(t);
      }
    }

    let bestThreshold = thresholds[0];
    let bestFnRate = 1.0;
    let bestFpRate = 0.0;

    for (const threshold of thresholds) {
      const { fnRate, fpRate } = this.evaluateThreshold(scores, labels, threshold);
      const cost = this.costPositive * fnRate + this.costNegative * fpRate;

      if (cost < this.costPositive * bestFnRate + this.costNegative * bestFpRate) {
        bestThreshold = threshold;
        bestFnRate = fnRate;
        bestFpRate = fpRate;
      }
    }

    const tp = scores.filter((s, i) => s >= bestThreshold && labels[i] === 1).length;
    const fn = scores.filter((s, i) => s < bestThreshold && labels[i] === 1).length;
    const fp = scores.filter((s, i) => s >= bestThreshold && labels[i] === 0).length;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;

    return {
      threshold: bestThreshold,
      fnRate: bestFnRate,
      fpRate: bestFpRate,
      recall,
      precision,
    };
  }

  private evaluateThreshold(scores: number[], labels: number[], threshold: number): {
    fnRate: number;
    fpRate: number;
  } {
    const tp = scores.filter((s, i) => s >= threshold && labels[i] === 1).length;
    const tn = scores.filter((s, i) => s < threshold && labels[i] === 0).length;
    const fp = scores.filter((s, i) => s >= threshold && labels[i] === 0).length;
    const fn = scores.filter((s, i) => s < threshold && labels[i] === 1).length;

    const fnRate = tp + fn > 0 ? fn / (tp + fn) : 0;
    const fpRate = fp + tn > 0 ? fp / (fp + tn) : 0;

    return { fnRate, fpRate };
  }
}
