/**
 * 🔱 ZKAEDI PRIME — Enhanced Bayesian Optimization
 * 
 * ZKAEDI PRIME recursive Hamiltonian dynamics for optimization
 */

import { BayesianOptimizer, type BayesianOptimizerOptions } from "./bayesianOptimization.js";

export interface ZkaediPrimeBOOptions extends BayesianOptimizerOptions {
  eta?: number;
  gamma?: number;
  beta?: number;
  sigma?: number;
}

/**
 * ZKAEDI PRIME enhanced Bayesian Optimizer
 */
export class ZkaediPrimeBO extends BayesianOptimizer {
  private eta: number;
  private gamma: number;
  private beta: number;
  private sigma: number;
  private H: number = 0;

  constructor(
    bounds: Record<string, [number, number]>,
    options: ZkaediPrimeBOOptions = {}
  ) {
    super(bounds, options);
    this.eta = options.eta ?? 0.4;
    this.gamma = options.gamma ?? 0.3;
    this.beta = options.beta ?? 0.1;
    this.sigma = options.sigma ?? 0.05;
  }

  /**
   * ZKAEDI PRIME recursive Hamiltonian update
   */
  protected updateHamiltonian(y: number): void {
    const H_base = -y; // Base Hamiltonian (negative for minimization)
    const sigmoid = 1 / (1 + Math.exp(-this.gamma * this.H));
    const noise = this.gaussianNoise(0, 1 + this.beta * Math.abs(this.H));

    this.H = H_base + this.eta * this.H * sigmoid + this.sigma * noise;
  }

  private gaussianNoise(mean: number, std: number): number {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  }
}
