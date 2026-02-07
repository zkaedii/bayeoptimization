/**
 * 🔱 ZKAEDI PRIME — Confusion Matrix Defense
 * 
 * Optimize FP/FN rates for high-stakes detection systems
 */

export interface ConfusionMatrixDefenseOptions {
  fpBudget?: number;
  fnBudget?: number;
}

export interface ConfusionMatrixMetrics {
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
  falsePositiveRate: number;
  falseNegativeRate: number;
  precision: number;
  recall: number;
  f1Score: number;
}

/**
 * Confusion Matrix Defense for edit accuracy optimization
 */
export class ConfusionMatrixDefense {
  private fpBudget: number;
  private fnBudget: number;
  private tp: number = 0;
  private tn: number = 0;
  private fp: number = 0;
  private fn: number = 0;

  constructor(options: ConfusionMatrixDefenseOptions = {}) {
    this.fpBudget = options.fpBudget ?? 0.05; // Max 5% FP
    this.fnBudget = options.fnBudget ?? 0.01; // Max 1% FN
  }

  /**
   * Update confusion matrix
   */
  update(predicted: 0 | 1, trueLabel: 0 | 1, _confidence: number): void {
    if (predicted === 1 && trueLabel === 1) {
      this.tp++;
    } else if (predicted === 0 && trueLabel === 0) {
      this.tn++;
    } else if (predicted === 1 && trueLabel === 0) {
      this.fp++;
    } else if (predicted === 0 && trueLabel === 1) {
      this.fn++;
    }
  }

  /**
   * Compute metrics
   */
  computeMetrics(): ConfusionMatrixMetrics {
    const total = this.tp + this.tn + this.fp + this.fn;
    const fpRate = total > 0 ? this.fp / (this.fp + this.tn) : 0;
    const fnRate = total > 0 ? this.fn / (this.fn + this.tp) : 0;
    const precision = this.tp + this.fp > 0 ? this.tp / (this.tp + this.fp) : 0;
    const recall = this.tp + this.fn > 0 ? this.tp / (this.tp + this.fn) : 0;
    const f1Score = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      truePositives: this.tp,
      trueNegatives: this.tn,
      falsePositives: this.fp,
      falseNegatives: this.fn,
      falsePositiveRate: fpRate,
      falseNegativeRate: fnRate,
      precision,
      recall,
      f1Score,
    };
  }

  /**
   * Check if within budget
   */
  isWithinBudget(): boolean {
    const metrics = this.computeMetrics();
    return metrics.falsePositiveRate <= this.fpBudget && metrics.falseNegativeRate <= this.fnBudget;
  }
}
