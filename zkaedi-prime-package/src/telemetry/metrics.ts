/**
 * 🔱 ZKAEDI PRIME — Standard Metrics Definitions
 * 
 * Define standard metrics for optimization and ML operations
 */

export interface MetricDefinition {
  name: string;
  type: 'counter' | 'gauge' | 'histogram';
  help: string;
  labelNames?: string[];
  buckets?: number[];
}

/**
 * Standard metrics for ZKAEDI PRIME
 */
export const STANDARD_METRICS: Record<string, MetricDefinition> = {
  // Optimization metrics
  optimizationIterationsTotal: {
    name: 'zkaedi_optimization_iterations_total',
    type: 'counter',
    help: 'Total number of optimization iterations',
    labelNames: ['status'],
  },
  optimizationObjectiveValue: {
    name: 'zkaedi_optimization_objective_value',
    type: 'histogram',
    help: 'Distribution of objective function values',
    buckets: [-100, -10, -1, 0, 1, 10, 100],
  },
  gpPredictionLatencyMs: {
    name: 'zkaedi_gp_prediction_latency_ms',
    type: 'histogram',
    help: 'Gaussian Process prediction latency in milliseconds',
    buckets: [1, 5, 10, 50, 100, 500, 1000],
  },
  acquisitionFunctionEvaluationTimeMs: {
    name: 'zkaedi_acquisition_function_evaluation_time_ms',
    type: 'histogram',
    help: 'Acquisition function evaluation time in milliseconds',
    buckets: [1, 10, 100, 500, 1000, 5000],
  },
  modelTrainingTimeMs: {
    name: 'zkaedi_model_training_time_ms',
    type: 'histogram',
    help: 'Model training time in milliseconds',
    buckets: [10, 50, 100, 500, 1000, 5000, 10000],
  },
  memoryUsageBytes: {
    name: 'zkaedi_memory_usage_bytes',
    type: 'gauge',
    help: 'Memory usage in bytes',
  },
  activeOptimizations: {
    name: 'zkaedi_active_optimizations',
    type: 'gauge',
    help: 'Number of currently active optimizations',
  },
};

/**
 * Get metric definition by name
 */
export function getMetricDefinition(name: string): MetricDefinition | undefined {
  return STANDARD_METRICS[name];
}

/**
 * Get all metric definitions
 */
export function getAllMetricDefinitions(): MetricDefinition[] {
  return Object.values(STANDARD_METRICS);
}
