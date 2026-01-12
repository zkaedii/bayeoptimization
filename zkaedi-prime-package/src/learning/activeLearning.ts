/**
 * 🔱 ZKAEDI PRIME — Active Learning
 * 
 * Intelligent query selection with uncertainty and diversity
 */

export interface ActiveLearningOptions {
  wUncertainty?: number;
  wDiversity?: number;
}

export interface ActiveLearningResult {
  selectedIndices: number[];
  scores: number[];
}

/**
 * Active Learning with Hamiltonian Query Selection
 */
export class ZkaediPrimeActiveLearning {
  private wUncertainty: number;
  private wDiversity: number;

  constructor(options: ActiveLearningOptions = {}) {
    this.wUncertainty = options.wUncertainty ?? 0.6;
    this.wDiversity = options.wDiversity ?? 0.1;
  }

  /**
   * Select batch of samples for annotation
   */
  selectBatch(
    unlabeledData: any[],
    model: {
      predictUncertainty: (data: any) => number;
      extractFeatures?: (data: any) => number[];
    },
    budget: number
  ): ActiveLearningResult {
    const scores: number[] = [];
    const features = unlabeledData.map((d) =>
      model.extractFeatures ? model.extractFeatures(d) : []
    );

    // Compute scores for each sample
    for (let i = 0; i < unlabeledData.length; i++) {
      const uncertainty = model.predictUncertainty(unlabeledData[i]);
      const diversity = this.computeDiversity(i, features, []);
      const score = this.wUncertainty * uncertainty + this.wDiversity * diversity;
      scores.push(score);
    }

    // Select top-k
    const indexed = scores.map((score, idx) => ({ score, idx }));
    indexed.sort((a, b) => b.score - a.score);
    const selectedIndices = indexed.slice(0, budget).map((item) => item.idx);

    return {
      selectedIndices,
      scores,
    };
  }

  private computeDiversity(
    idx: number,
    features: number[][],
    selected: number[]
  ): number {
    if (selected.length === 0) return 1.0;

    const currentFeatures = features[idx];
    let minDistance = Infinity;

    for (const selIdx of selected) {
      const selFeatures = features[selIdx];
      const distance = this.euclideanDistance(currentFeatures, selFeatures);
      minDistance = Math.min(minDistance, distance);
    }

    return minDistance;
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
