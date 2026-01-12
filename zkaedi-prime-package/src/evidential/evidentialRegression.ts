/**
 * 🔱 ZKAEDI PRIME — Evidential Regression
 * 
 * Evidential Deep Learning for regression with uncertainty quantification
 */

export interface EvidentialRegressorOptions {
  klWeight?: number;
}

export interface EvidentialRegressionOutput {
  mu: number;
  sigma: number;
  nu: number;
  alpha: number;
  aleatoricUncertainty: number;
  epistemicUncertainty: number;
  totalUncertainty: number;
}

/**
 * Evidential Regressor using Normal-Inverse-Gamma distribution
 */
export class EvidentialRegressor {
  private klWeight: number;

  constructor(options: EvidentialRegressorOptions = {}) {
    this.klWeight = options.klWeight ?? 0.001;
  }

  /**
   * Forward pass: convert network outputs to NIG parameters
   */
  forward(outputs: [number, number, number, number]): EvidentialRegressionOutput {
    const [mu, logSigma, logNu, logAlpha] = outputs;

    const sigma = Math.exp(logSigma);
    const nu = Math.exp(logNu);
    const alpha = Math.exp(logAlpha);

    // Aleatoric uncertainty: sigma^2
    const aleatoricUncertainty = sigma * sigma;

    // Epistemic uncertainty: beta / (nu * (alpha - 1))
    // where beta = nu * sigma^2 (simplified)
    const beta = nu * aleatoricUncertainty;
    const epistemicUncertainty = beta / (nu * Math.max(alpha - 1, 1e-6));

    // Total uncertainty
    const totalUncertainty = aleatoricUncertainty + epistemicUncertainty;

    return {
      mu,
      sigma,
      nu,
      alpha,
      aleatoricUncertainty,
      epistemicUncertainty,
      totalUncertainty,
    };
  }
}
