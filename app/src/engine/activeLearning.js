/**
 * ZKAEDI PRIME - Active Learning Engine
 * Uncertainty-driven and diversity-based sample selection
 */

class ActiveLearningSimulator {
  constructor(options = {}) {
    this.nClasses = options.nClasses || 3;
    this.uncertaintyWeight = options.uncertaintyWeight || 0.6;
    this.diversityWeight = options.diversityWeight || 0.4;
    this.labeledPool = [];
    this.unlabeledPool = [];
    this.history = [];
    this.accuracyHistory = [];
    this.iteration = 0;
  }

  // Generate a synthetic 2D classification dataset
  generateDataset(nSamples = 200) {
    this.labeledPool = [];
    this.unlabeledPool = [];
    this.history = [];
    this.accuracyHistory = [];
    this.iteration = 0;

    const centers = [];
    for (let c = 0; c < this.nClasses; c++) {
      const angle = (2 * Math.PI * c) / this.nClasses;
      centers.push([Math.cos(angle) * 3, Math.sin(angle) * 3]);
    }

    for (let i = 0; i < nSamples; i++) {
      const classIdx = Math.floor(Math.random() * this.nClasses);
      const center = centers[classIdx];
      const x = center[0] + (Math.random() - 0.5) * 3;
      const y = center[1] + (Math.random() - 0.5) * 3;

      this.unlabeledPool.push({
        id: i,
        features: [x, y],
        trueLabel: classIdx,
        labeled: false
      });
    }

    // Seed with a few labeled points
    const seedIndices = [];
    for (let c = 0; c < this.nClasses; c++) {
      const classPoints = this.unlabeledPool.filter(p => p.trueLabel === c);
      if (classPoints.length > 0) {
        seedIndices.push(classPoints[0].id);
      }
    }

    seedIndices.forEach(id => {
      const idx = this.unlabeledPool.findIndex(p => p.id === id);
      if (idx >= 0) {
        const point = this.unlabeledPool.splice(idx, 1)[0];
        point.labeled = true;
        this.labeledPool.push(point);
      }
    });

    return this.getState();
  }

  // Simple kNN-based uncertainty estimation
  _estimateUncertainty(point) {
    if (this.labeledPool.length === 0) return 1.0;

    const k = Math.min(5, this.labeledPool.length);
    const distances = this.labeledPool.map(lp => ({
      dist: Math.sqrt(
        (lp.features[0] - point.features[0]) ** 2 +
        (lp.features[1] - point.features[1]) ** 2
      ),
      label: lp.trueLabel
    }));

    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, k);

    // Count class votes
    const votes = new Float64Array(this.nClasses);
    const totalWeight = neighbors.reduce((s, n) => s + 1 / (n.dist + 0.1), 0);
    neighbors.forEach(n => {
      votes[n.label] += (1 / (n.dist + 0.1)) / totalWeight;
    });

    // Entropy as uncertainty
    let entropy = 0;
    for (let c = 0; c < this.nClasses; c++) {
      if (votes[c] > 0) {
        entropy -= votes[c] * Math.log2(votes[c] + 1e-10);
      }
    }

    return entropy / Math.log2(this.nClasses); // Normalized [0,1]
  }

  // Diversity: distance to nearest labeled point
  _estimateDiversity(point) {
    if (this.labeledPool.length === 0) return 1.0;

    let minDist = Infinity;
    for (const lp of this.labeledPool) {
      const dist = Math.sqrt(
        (lp.features[0] - point.features[0]) ** 2 +
        (lp.features[1] - point.features[1]) ** 2
      );
      minDist = Math.min(minDist, dist);
    }

    // Normalize
    return Math.min(1.0, minDist / 5);
  }

  // Select next batch of points to label
  selectBatch(batchSize = 5) {
    if (this.unlabeledPool.length === 0) return null;

    const scored = this.unlabeledPool.map(point => {
      const uncertainty = this._estimateUncertainty(point);
      const diversity = this._estimateDiversity(point);
      const score = this.uncertaintyWeight * uncertainty + this.diversityWeight * diversity;

      return { point, uncertainty, diversity, score };
    });

    scored.sort((a, b) => b.score - a.score);
    const selected = scored.slice(0, Math.min(batchSize, scored.length));

    // Move selected to labeled pool
    const selectedIds = new Set(selected.map(s => s.point.id));
    const newLabeled = [];
    this.unlabeledPool = this.unlabeledPool.filter(p => {
      if (selectedIds.has(p.id)) {
        p.labeled = true;
        this.labeledPool.push(p);
        newLabeled.push(p);
        return false;
      }
      return true;
    });

    // Compute accuracy on all data
    const accuracy = this._computeAccuracy();
    this.iteration++;

    const stepResult = {
      iteration: this.iteration,
      selectedPoints: selected.map(s => ({
        id: s.point.id,
        features: s.point.features,
        trueLabel: s.point.trueLabel,
        uncertainty: s.uncertainty,
        diversity: s.diversity,
        score: s.score
      })),
      labeledCount: this.labeledPool.length,
      unlabeledCount: this.unlabeledPool.length,
      accuracy,
      uncertaintyScores: scored.map(s => ({
        features: s.point.features,
        uncertainty: s.uncertainty,
        diversity: s.diversity
      }))
    };

    this.history.push(stepResult);
    this.accuracyHistory.push({ iteration: this.iteration, accuracy, nLabeled: this.labeledPool.length });

    return stepResult;
  }

  _computeAccuracy() {
    const allPoints = [...this.labeledPool, ...this.unlabeledPool];
    let correct = 0;

    for (const point of allPoints) {
      const k = Math.min(3, this.labeledPool.length);
      const distances = this.labeledPool.map(lp => ({
        dist: Math.sqrt(
          (lp.features[0] - point.features[0]) ** 2 +
          (lp.features[1] - point.features[1]) ** 2
        ),
        label: lp.trueLabel
      }));
      distances.sort((a, b) => a.dist - b.dist);
      const neighbors = distances.slice(0, k);

      const votes = new Float64Array(this.nClasses);
      neighbors.forEach(n => { votes[n.label]++; });

      const predicted = votes.indexOf(Math.max(...votes));
      if (predicted === point.trueLabel) correct++;
    }

    return correct / allPoints.length;
  }

  getState() {
    return {
      labeledPool: this.labeledPool.map(p => ({
        id: p.id,
        features: p.features,
        trueLabel: p.trueLabel
      })),
      unlabeledPool: this.unlabeledPool.map(p => ({
        id: p.id,
        features: p.features
      })),
      iteration: this.iteration,
      accuracyHistory: this.accuracyHistory,
      nLabeled: this.labeledPool.length,
      nUnlabeled: this.unlabeledPool.length
    };
  }
}

module.exports = { ActiveLearningSimulator };
