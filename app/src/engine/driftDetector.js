/**
 * ZKAEDI PRIME - Temporal Drift Detection Engine
 * Sliding window KL divergence with phase detection
 */

class DriftDetector {
  constructor(options = {}) {
    this.windowSize = options.windowSize || 20;
    this.driftThreshold = options.driftThreshold || 0.3;
    this.criticalThreshold = options.criticalThreshold || 0.6;
    this.nBins = options.nBins || 10;

    this.referenceWindow = [];
    this.currentWindow = [];
    this.history = [];
    this.phase = 'STABLE';
    this.tick = 0;
  }

  _buildHistogram(values) {
    if (values.length === 0) return new Float64Array(this.nBins).fill(1 / this.nBins);

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const bins = new Float64Array(this.nBins);

    values.forEach(v => {
      let idx = Math.floor(((v - min) / range) * (this.nBins - 1));
      idx = Math.max(0, Math.min(this.nBins - 1, idx));
      bins[idx]++;
    });

    // Normalize + Laplace smoothing
    const total = values.length + this.nBins * 0.01;
    for (let i = 0; i < this.nBins; i++) {
      bins[i] = (bins[i] + 0.01) / total;
    }

    return bins;
  }

  _klDivergence(p, q) {
    let kl = 0;
    for (let i = 0; i < p.length; i++) {
      if (p[i] > 0) {
        kl += p[i] * Math.log(p[i] / q[i]);
      }
    }
    return kl;
  }

  _symmetricKL(p, q) {
    return (this._klDivergence(p, q) + this._klDivergence(q, p)) / 2;
  }

  addObservation(value) {
    this.tick++;

    if (this.referenceWindow.length < this.windowSize) {
      this.referenceWindow.push(value);
      this.history.push({
        tick: this.tick,
        value,
        klDivergence: 0,
        phase: 'WARMUP',
        driftScore: 0,
        referenceWindowSize: this.referenceWindow.length,
        currentWindowSize: 0
      });
      return this.history[this.history.length - 1];
    }

    this.currentWindow.push(value);
    if (this.currentWindow.length > this.windowSize) {
      this.currentWindow.shift();
    }

    if (this.currentWindow.length < 3) {
      this.history.push({
        tick: this.tick,
        value,
        klDivergence: 0,
        phase: 'STABLE',
        driftScore: 0,
        referenceWindowSize: this.referenceWindow.length,
        currentWindowSize: this.currentWindow.length
      });
      return this.history[this.history.length - 1];
    }

    const refHist = this._buildHistogram(this.referenceWindow);
    const curHist = this._buildHistogram(this.currentWindow);
    const kl = this._symmetricKL(refHist, curHist);

    // Determine phase
    let phase;
    if (kl >= this.criticalThreshold) {
      phase = 'CRITICAL';
    } else if (kl >= this.driftThreshold) {
      phase = 'DRIFT';
    } else {
      phase = 'STABLE';
    }

    // Adaptive reference window update (only in STABLE phase)
    if (phase === 'STABLE' && this.currentWindow.length >= this.windowSize) {
      this.referenceWindow = [...this.currentWindow];
      this.currentWindow = [];
    }

    this.phase = phase;

    const entry = {
      tick: this.tick,
      value,
      klDivergence: kl,
      phase,
      driftScore: Math.min(1, kl / this.criticalThreshold),
      referenceWindowSize: this.referenceWindow.length,
      currentWindowSize: this.currentWindow.length,
      referenceMean: this.referenceWindow.reduce((a, b) => a + b, 0) / this.referenceWindow.length,
      currentMean: this.currentWindow.reduce((a, b) => a + b, 0) / this.currentWindow.length
    };

    this.history.push(entry);
    return entry;
  }

  // Generate a full synthetic scenario
  generateScenario(nTicks = 150) {
    this.referenceWindow = [];
    this.currentWindow = [];
    this.history = [];
    this.phase = 'STABLE';
    this.tick = 0;

    const results = [];

    for (let t = 0; t < nTicks; t++) {
      let value;

      if (t < 50) {
        // Stable phase: normal distribution centered at 5
        value = 5 + (Math.random() - 0.5) * 2;
      } else if (t < 80) {
        // Gradual drift: mean shifts from 5 to 8
        const progress = (t - 50) / 30;
        const mean = 5 + progress * 3;
        value = mean + (Math.random() - 0.5) * 2;
      } else if (t < 100) {
        // Drift stabilized at new distribution
        value = 8 + (Math.random() - 0.5) * 2;
      } else if (t < 110) {
        // Sudden shift (concept drift)
        value = 2 + (Math.random() - 0.5) * 4;
      } else {
        // Recovery to original
        const progress = Math.min(1, (t - 110) / 20);
        const mean = 2 + progress * 3;
        value = mean + (Math.random() - 0.5) * 2;
      }

      results.push(this.addObservation(value));
    }

    return results;
  }

  getState() {
    return {
      phase: this.phase,
      tick: this.tick,
      history: this.history,
      config: {
        windowSize: this.windowSize,
        driftThreshold: this.driftThreshold,
        criticalThreshold: this.criticalThreshold
      }
    };
  }
}

module.exports = { DriftDetector };
