/**
 * ZKAEDI PRIME - Bayesian Optimization Engine
 * Implements Gaussian Process surrogate + acquisition functions
 * with Hamiltonian dynamics enhancement
 */

class GaussianProcess {
  constructor(kernel = 'rbf', lengthScale = 1.0, noise = 1e-6) {
    this.kernel = kernel;
    this.lengthScale = lengthScale;
    this.noise = noise;
    this.X = [];
    this.y = [];
    this.K_inv = null;
    this.alpha = null;
  }

  rbfKernel(x1, x2) {
    let sq = 0;
    for (let i = 0; i < x1.length; i++) {
      sq += (x1[i] - x2[i]) ** 2;
    }
    return Math.exp(-sq / (2 * this.lengthScale ** 2));
  }

  maternKernel(x1, x2) {
    let r = 0;
    for (let i = 0; i < x1.length; i++) {
      r += (x1[i] - x2[i]) ** 2;
    }
    r = Math.sqrt(r) / this.lengthScale;
    const sqrt5r = Math.sqrt(5) * r;
    return (1 + sqrt5r + (5 * r * r) / 3) * Math.exp(-sqrt5r);
  }

  computeKernel(x1, x2) {
    return this.kernel === 'matern' ? this.maternKernel(x1, x2) : this.rbfKernel(x1, x2);
  }

  fit(X, y) {
    this.X = X.map(x => [...x]); // deep copy
    this.y = [...y];
    const n = X.length;

    // Build kernel matrix
    const K = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const k = this.computeKernel(X[i], X[j]);
        K[i][j] = k + (i === j ? this.noise : 0);
        K[j][i] = K[i][j];
      }
    }

    // Cholesky-like solve via regularized inversion
    this.K_inv = this._invertMatrix(K, n);
    this.alpha = this._matVecMul(this.K_inv, y, n);
  }

  predict(xStar) {
    const n = this.X.length;
    if (n === 0 || !this.K_inv || !this.alpha) return { mean: 0, variance: 1.0 };

    // k* vector
    const kStar = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      kStar[i] = this.computeKernel(xStar, this.X[i]);
    }

    // mean = k*^T * alpha
    let mean = 0;
    for (let i = 0; i < n; i++) {
      mean += kStar[i] * this.alpha[i];
    }

    // variance = k(x*,x*) - k*^T * K_inv * k*
    const kss = this.computeKernel(xStar, xStar) + this.noise;
    const v = this._matVecMul(this.K_inv, kStar, n);
    let variance = kss;
    for (let i = 0; i < n; i++) {
      variance -= kStar[i] * v[i];
    }
    variance = Math.max(variance, 1e-10);

    return { mean, variance };
  }

  _invertMatrix(M, n) {
    // Gaussian elimination with partial pivoting
    const aug = Array.from({ length: n }, (_, i) => {
      const row = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) row[j] = M[i][j];
      row[n + i] = 1;
      return row;
    });

    for (let col = 0; col < n; col++) {
      let maxVal = Math.abs(aug[col][col]);
      let maxRow = col;
      for (let row = col + 1; row < n; row++) {
        if (Math.abs(aug[row][col]) > maxVal) {
          maxVal = Math.abs(aug[row][col]);
          maxRow = row;
        }
      }
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

      const pivot = aug[col][col];
      if (Math.abs(pivot) < 1e-12) {
        aug[col][col] += 1e-6; // Regularize
      }

      const invPivot = 1.0 / aug[col][col];
      for (let j = 0; j < 2 * n; j++) aug[col][j] *= invPivot;

      for (let row = 0; row < n; row++) {
        if (row === col) continue;
        const factor = aug[row][col];
        for (let j = 0; j < 2 * n; j++) {
          aug[row][j] -= factor * aug[col][j];
        }
      }
    }

    return aug.map(row => Array.from(row.slice(n)));
  }

  _matVecMul(M, v, n) {
    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (!M[i]) continue;
      for (let j = 0; j < n; j++) {
        result[i] += (M[i][j] || 0) * (v[j] || 0);
      }
    }
    return result;
  }
}

class BayesianOptimizer {
  constructor(bounds, options = {}) {
    this.bounds = bounds; // [[low, high], [low, high], ...]
    this.nDims = bounds.length;
    this.nIter = options.nIter || 50;
    this.nWarmup = options.nWarmup || 5;
    this.acquisitionFn = options.acquisitionFunction || 'EI';
    this.kernel = options.kernel || 'rbf';
    this.noise = options.noise || 1e-4;
    this.eta = options.eta || 0.1;     // Hamiltonian coupling
    this.gamma = options.gamma || 0.95; // Momentum decay
    this.kappa = options.kappa || 2.576; // UCB exploration param

    this.X = [];
    this.y = [];
    this.bestX = null;
    this.bestY = -Infinity;
    this.trajectory = [];
    this.gp = new GaussianProcess(this.kernel, 1.0, this.noise);

    // Hamiltonian state
    this.momentum = new Float64Array(this.nDims);
    this.hamiltonianEnergy = 0;
  }

  _randomSample() {
    return this.bounds.map(([lo, hi]) => lo + Math.random() * (hi - lo));
  }

  _clamp(x) {
    return x.map((val, i) => Math.max(this.bounds[i][0], Math.min(this.bounds[i][1], val)));
  }

  _normalCDF(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804014327;
    const p = d * Math.exp(-x * x / 2) *
      (t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.3302744)))));
    return x > 0 ? 1 - p : p;
  }

  _normalPDF(x) {
    return Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
  }

  _acquisitionEI(mean, variance) {
    if (variance < 1e-10) return 0;
    const std = Math.sqrt(variance);
    const z = (mean - this.bestY) / std;
    return (mean - this.bestY) * this._normalCDF(z) + std * this._normalPDF(z);
  }

  _acquisitionPI(mean, variance) {
    if (variance < 1e-10) return 0;
    const z = (mean - this.bestY) / Math.sqrt(variance);
    return this._normalCDF(z);
  }

  _acquisitionUCB(mean, variance) {
    return mean + this.kappa * Math.sqrt(variance);
  }

  _evaluate(x) {
    const { mean, variance } = this.gp.predict(x);
    switch (this.acquisitionFn) {
      case 'PI': return this._acquisitionPI(mean, variance);
      case 'UCB': return this._acquisitionUCB(mean, variance);
      default: return this._acquisitionEI(mean, variance);
    }
  }

  _hamiltonianStep(x) {
    // Update momentum with Hamiltonian dynamics
    const grad = this._numericalGradient(x);
    for (let i = 0; i < this.nDims; i++) {
      this.momentum[i] = this.gamma * this.momentum[i] + this.eta * grad[i];
    }

    // Propose new point
    const proposed = x.map((val, i) => val + this.momentum[i]);
    return this._clamp(proposed);
  }

  _numericalGradient(x, eps = 0.01) {
    const grad = new Float64Array(this.nDims);
    const f0 = this._evaluate(x);
    for (let i = 0; i < this.nDims; i++) {
      const xp = [...x];
      xp[i] += eps;
      xp[i] = Math.min(this.bounds[i][1], xp[i]);
      grad[i] = (this._evaluate(xp) - f0) / eps;
    }
    return grad;
  }

  _maximizeAcquisition() {
    let bestAcq = -Infinity;
    let bestPoint = null;

    // Multi-start gradient ascent with Hamiltonian dynamics
    const nRestarts = 20;
    for (let r = 0; r < nRestarts; r++) {
      let x = this._randomSample();
      this.momentum.fill(0);

      for (let step = 0; step < 15; step++) {
        x = this._hamiltonianStep(x);
      }

      const acq = this._evaluate(x);
      if (acq > bestAcq) {
        bestAcq = acq;
        bestPoint = [...x];
      }
    }

    // Also try random candidates
    for (let i = 0; i < 100; i++) {
      const x = this._randomSample();
      const acq = this._evaluate(x);
      if (acq > bestAcq) {
        bestAcq = acq;
        bestPoint = [...x];
      }
    }

    return bestPoint;
  }

  async optimize(objectiveFn, onStep = null) {
    this.X = [];
    this.y = [];
    this.bestY = -Infinity;
    this.trajectory = [];

    // Warmup phase: random sampling
    for (let i = 0; i < this.nWarmup; i++) {
      const x = this._randomSample();
      const y = await objectiveFn(x);
      this.X.push(x);
      this.y.push(y);

      if (y > this.bestY) {
        this.bestY = y;
        this.bestX = [...x];
      }

      const step = {
        iteration: i,
        phase: 'warmup',
        x: [...x],
        y,
        bestY: this.bestY,
        bestX: [...this.bestX],
        acquisitionValue: 0,
        gpMean: 0,
        gpVariance: 1,
        hamiltonianEnergy: 0
      };
      this.trajectory.push(step);
      if (onStep) onStep(step);
    }

    // Optimization phase
    for (let i = 0; i < this.nIter; i++) {
      this.gp.fit(this.X, this.y);
      const nextX = this._maximizeAcquisition();
      const nextY = await objectiveFn(nextX);
      const { mean, variance } = this.gp.predict(nextX);

      this.X.push(nextX);
      this.y.push(nextY);

      if (nextY > this.bestY) {
        this.bestY = nextY;
        this.bestX = [...nextX];
      }

      // Compute Hamiltonian energy
      let kineticEnergy = 0;
      for (let d = 0; d < this.nDims; d++) {
        kineticEnergy += this.momentum[d] ** 2;
      }
      this.hamiltonianEnergy = -nextY + 0.5 * kineticEnergy;

      const step = {
        iteration: this.nWarmup + i,
        phase: 'optimization',
        x: [...nextX],
        y: nextY,
        bestY: this.bestY,
        bestX: [...this.bestX],
        acquisitionValue: this._evaluate(nextX),
        gpMean: mean,
        gpVariance: variance,
        hamiltonianEnergy: this.hamiltonianEnergy
      };
      this.trajectory.push(step);
      if (onStep) onStep(step);
    }

    return {
      bestX: this.bestX,
      bestY: this.bestY,
      nIter: this.nWarmup + this.nIter,
      trajectory: this.trajectory
    };
  }

  // Generate GP surface for visualization
  getSurface(resolution = 30) {
    if (this.X.length < 2 || this.nDims !== 2) return null;

    this.gp.fit(this.X, this.y);
    const [xBounds, yBounds] = this.bounds;
    const xStep = (xBounds[1] - xBounds[0]) / (resolution - 1);
    const yStep = (yBounds[1] - yBounds[0]) / (resolution - 1);

    const surface = { x: [], y: [], mean: [], variance: [], acquisition: [] };

    for (let i = 0; i < resolution; i++) {
      const xRow = [], yRow = [], meanRow = [], varRow = [], acqRow = [];
      for (let j = 0; j < resolution; j++) {
        const px = xBounds[0] + j * xStep;
        const py = yBounds[0] + i * yStep;
        const { mean, variance } = this.gp.predict([px, py]);
        xRow.push(px);
        yRow.push(py);
        meanRow.push(mean);
        varRow.push(variance);
        acqRow.push(this._evaluate([px, py]));
      }
      surface.x.push(xRow);
      surface.y.push(yRow);
      surface.mean.push(meanRow);
      surface.variance.push(varRow);
      surface.acquisition.push(acqRow);
    }

    return surface;
  }
}

module.exports = { BayesianOptimizer, GaussianProcess };
