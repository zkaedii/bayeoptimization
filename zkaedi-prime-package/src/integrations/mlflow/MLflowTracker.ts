/**
 * 🔱 ZKAEDI PRIME — MLflow Tracker
 * 
 * High-level MLflow tracking interface for Bayesian Optimization
 */

import { MLflowClient, MLflowRun } from './MLflowClient';
import { MLflowConfig } from '../../config';

/**
 * MLflow Tracker for experiment tracking
 */
export class MLflowTracker {
  private client: MLflowClient;
  private config: MLflowConfig;
  private currentExperimentId?: string;
  private currentRun?: MLflowRun;

  constructor(config: MLflowConfig = {}) {
    this.config = config;
    this.client = new MLflowClient(config);
  }

  /**
   * Create a new experiment
   */
  async createExperiment(name: string, description?: string): Promise<string> {
    const experimentId = await this.client.createExperiment(name);
    this.currentExperimentId = experimentId;

    if (description) {
      // Description would be set as a tag on runs within this experiment
    }

    return experimentId;
  }

  /**
   * Set the current experiment
   */
  async setExperiment(experimentIdOrName: string): Promise<void> {
    // Try to get experiment by ID first, then by name
    try {
      this.currentExperimentId = experimentIdOrName;
    } catch {
      const experiment = await this.client.getExperimentByName(experimentIdOrName);
      this.currentExperimentId = experiment.experiment_id;
    }
  }

  /**
   * Start a new run
   */
  async startRun(runName?: string, tags?: Record<string, string>): Promise<string> {
    if (!this.currentExperimentId) {
      // Create default experiment
      const experimentName = this.config.experimentName ?? 'zkaedi-prime';
      this.currentExperimentId = await this.client.createExperiment(experimentName);
    }

    this.currentRun = await this.client.createRun(this.currentExperimentId, runName, tags);
    return this.currentRun.run_id;
  }

  /**
   * End the current run
   */
  async endRun(status: 'FINISHED' | 'FAILED' | 'KILLED' = 'FINISHED'): Promise<void> {
    if (!this.currentRun) {
      throw new Error('No active run to end');
    }

    await this.client.updateRun(this.currentRun.run_id, status);
    this.currentRun = undefined;
  }

  /**
   * Log a metric
   */
  async logMetric(key: string, value: number, step?: number, timestamp?: number): Promise<void> {
    if (!this.currentRun) {
      throw new Error('No active run. Call startRun() first.');
    }

    await this.client.logMetric(this.currentRun.run_id, key, value, timestamp, step);
  }

  /**
   * Log multiple metrics
   */
  async logMetrics(metrics: Record<string, number>, step?: number): Promise<void> {
    if (!this.currentRun) {
      throw new Error('No active run. Call startRun() first.');
    }

    const metricsArray = Object.entries(metrics).map(([key, value]) => ({
      key,
      value,
      step,
      timestamp: Date.now(),
    }));

    await this.client.logBatch(this.currentRun.run_id, metricsArray);
  }

  /**
   * Log a parameter
   */
  async logParam(key: string, value: string | number): Promise<void> {
    if (!this.currentRun) {
      throw new Error('No active run. Call startRun() first.');
    }

    await this.client.logParam(this.currentRun.run_id, key, String(value));
  }

  /**
   * Log multiple parameters
   */
  async logParams(params: Record<string, any>): Promise<void> {
    if (!this.currentRun) {
      throw new Error('No active run. Call startRun() first.');
    }

    const paramsArray = Object.entries(params).map(([key, value]) => ({
      key,
      value: String(value),
    }));

    await this.client.logBatch(this.currentRun.run_id, undefined, paramsArray);
  }

  /**
   * Set a tag
   */
  async setTag(key: string, value: string): Promise<void> {
    if (!this.currentRun) {
      throw new Error('No active run. Call startRun() first.');
    }

    await this.client.setTag(this.currentRun.run_id, key, value);
  }

  /**
   * Set multiple tags
   */
  async setTags(tags: Record<string, string>): Promise<void> {
    if (!this.currentRun) {
      throw new Error('No active run. Call startRun() first.');
    }

    const tagsArray = Object.entries(tags).map(([key, value]) => ({ key, value }));
    await this.client.logBatch(this.currentRun.run_id, undefined, undefined, tagsArray);
  }

  /**
   * Log an optimization step (Bayesian Optimization specific)
   */
  async logOptimizationStep(
    iteration: number,
    params: number[],
    objective: number,
    metadata?: any
  ): Promise<void> {
    await this.logMetric('objective_value', objective, iteration);
    await this.logMetric('iteration', iteration, iteration);

    // Log parameters as metrics for visualization
    for (let i = 0; i < params.length; i++) {
      await this.logMetric(`param_${i}`, params[i], iteration);
    }

    // Log metadata if provided
    if (metadata) {
      for (const [key, value] of Object.entries(metadata)) {
        if (typeof value === 'number') {
          await this.logMetric(key, value, iteration);
        }
      }
    }
  }

  /**
   * Log GP hyperparameters
   */
  async logGPHyperparameters(lengthscale: number[], noise: number): Promise<void> {
    await this.logParam('gp_noise', noise);
    for (let i = 0; i < lengthscale.length; i++) {
      await this.logParam(`gp_lengthscale_${i}`, lengthscale[i]);
    }
  }

  /**
   * Get current run ID
   */
  getCurrentRunId(): string | undefined {
    return this.currentRun?.run_id;
  }

  /**
   * Check if there's an active run
   */
  hasActiveRun(): boolean {
    return this.currentRun !== undefined;
  }
}
