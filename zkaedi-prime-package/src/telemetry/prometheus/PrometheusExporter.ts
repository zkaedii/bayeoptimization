/**
 * 🔱 ZKAEDI PRIME — Prometheus Metrics Exporter
 * 
 * Export metrics in Prometheus format
 */

import * as promClient from 'prom-client';
import { PrometheusConfig } from '../../config';

/**
 * Prometheus Metrics Exporter
 */
export class PrometheusExporter {
  private register: promClient.Registry;
  private config: Required<PrometheusConfig>;

  // Standard metrics
  public optimizationDuration: promClient.Histogram;
  public optimizationIterations: promClient.Counter;
  public objectiveEvaluations: promClient.Counter;
  public gpTrainingTime: promClient.Histogram;
  public acquisitionFunctionTime: promClient.Histogram;
  public memoryUsage: promClient.Gauge;
  public errorRate: promClient.Counter;

  constructor(config: PrometheusConfig = {}) {
    this.config = {
      enabled: config.enabled ?? true,
      port: config.port ?? 9090,
      path: config.path ?? '/metrics',
    };

    this.register = new promClient.Registry();

    // Initialize standard metrics
    this.optimizationDuration = new promClient.Histogram({
      name: 'zkaedi_optimization_duration_seconds',
      help: 'Duration of optimization runs in seconds',
      labelNames: ['status'],
      buckets: [0.1, 0.5, 1, 5, 10, 30, 60, 120, 300],
      registers: [this.register],
    });

    this.optimizationIterations = new promClient.Counter({
      name: 'zkaedi_optimization_iterations_total',
      help: 'Total number of optimization iterations',
      labelNames: ['status'],
      registers: [this.register],
    });

    this.objectiveEvaluations = new promClient.Counter({
      name: 'zkaedi_objective_evaluations_total',
      help: 'Total number of objective function evaluations',
      registers: [this.register],
    });

    this.gpTrainingTime = new promClient.Histogram({
      name: 'zkaedi_gp_training_time_seconds',
      help: 'Gaussian Process training time in seconds',
      buckets: [0.001, 0.01, 0.1, 0.5, 1, 2, 5],
      registers: [this.register],
    });

    this.acquisitionFunctionTime = new promClient.Histogram({
      name: 'zkaedi_acquisition_function_time_seconds',
      help: 'Acquisition function evaluation time in seconds',
      buckets: [0.001, 0.01, 0.1, 0.5, 1],
      registers: [this.register],
    });

    this.memoryUsage = new promClient.Gauge({
      name: 'zkaedi_memory_usage_bytes',
      help: 'Memory usage in bytes',
      registers: [this.register],
    });

    this.errorRate = new promClient.Counter({
      name: 'zkaedi_errors_total',
      help: 'Total number of errors',
      labelNames: ['type'],
      registers: [this.register],
    });

    // Collect default metrics (optional)
    if (this.config.enabled) {
      promClient.collectDefaultMetrics({ register: this.register });
    }
  }

  /**
   * Get metrics in Prometheus text format
   */
  async getMetrics(): Promise<string> {
    return this.register.metrics();
  }

  /**
   * Reset all metrics
   */
  resetMetrics(): void {
    this.register.resetMetrics();
  }

  /**
   * Create a custom metric
   */
  createCustomMetric(
    name: string,
    type: 'counter' | 'gauge' | 'histogram',
    help: string,
    labelNames?: string[]
  ): promClient.Counter | promClient.Gauge | promClient.Histogram {
    switch (type) {
      case 'counter':
        return new promClient.Counter({
          name,
          help,
          labelNames,
          registers: [this.register],
        });
      case 'gauge':
        return new promClient.Gauge({
          name,
          help,
          labelNames,
          registers: [this.register],
        });
      case 'histogram':
        return new promClient.Histogram({
          name,
          help,
          labelNames,
          registers: [this.register],
        });
      default:
        throw new Error(`Unsupported metric type: ${type}`);
    }
  }

  /**
   * Get the registry
   */
  getRegister(): promClient.Registry {
    return this.register;
  }
}

/**
 * Global Prometheus exporter instance
 */
let globalExporter: PrometheusExporter | null = null;

/**
 * Get or create the global Prometheus exporter
 */
export function getPrometheusExporter(config?: PrometheusConfig): PrometheusExporter {
  if (!globalExporter) {
    globalExporter = new PrometheusExporter(config);
  }
  return globalExporter;
}

/**
 * Reset the global Prometheus exporter
 */
export function resetPrometheusExporter(): void {
  globalExporter = null;
}
