/**
 * 🔱 ZKAEDI PRIME — MLflow Tracking Example
 * 
 * Example demonstrating MLflow integration for experiment tracking
 */

import { BayesianOptimizer } from '../../src/optimization';
import { MLflowTracker } from '../../src/integrations/mlflow';

async function main() {
  console.log('🔱 ZKAEDI PRIME — MLflow Tracking Example\n');

  // Initialize MLflow tracker
  const tracker = new MLflowTracker({
    trackingUri: 'http://localhost:5000',
    experimentName: 'zkaedi-bayesian-optimization',
  });

  // Create experiment
  const experimentId = await tracker.createExperiment(
    'zkaedi-bayesian-optimization',
    'Bayesian optimization with MLflow tracking'
  );
  console.log(`✅ Created experiment: ${experimentId}`);

  // Start a run
  const runId = await tracker.startRun('optimization-run-1', {
    algorithm: 'bayesian',
    framework: 'zkaedi-prime',
  });
  console.log(`✅ Started run: ${runId}\n`);

  // Log configuration parameters
  await tracker.logParams({
    n_iter: 20,
    n_warmup: 5,
    acquisition_function: 'EI',
    kernel: 'RBF',
  });

  // Create Bayesian Optimizer
  const optimizer = new BayesianOptimizer(
    {
      x: [-5, 5],
      y: [-5, 5],
    },
    { nIter: 20, nWarmup: 5, verbose: true }
  );

  // Objective function: minimize Rosenbrock function
  const rosenbrock = ([x, y]: number[]): number => {
    const a = 1;
    const b = 100;
    return -((a - x) ** 2 + b * (y - x ** 2) ** 2); // Negative for minimization
  };

  console.log('🚀 Starting optimization...\n');

  // Track optimization progress
  let iteration = 0;
  const originalOptimize = optimizer.optimize.bind(optimizer);
  optimizer.optimize = async (objective: any) => {
    const result = await originalOptimize(objective);

    // Log each iteration
    for (const point of result.trajectory) {
      await tracker.logOptimizationStep(iteration++, point.x, point.y);
    }

    return result;
  };

  // Run optimization
  const result = await optimizer.optimize(rosenbrock);

  // Log final results
  await tracker.logMetrics({
    best_objective: result.bestY,
    total_iterations: result.nIter,
  });

  await tracker.logParams({
    best_x: result.bestX[0],
    best_y: result.bestX[1],
  });

  console.log(`\n✅ Optimization complete!`);
  console.log(`   Best parameters: [${result.bestX.join(', ')}]`);
  console.log(`   Best value: ${result.bestY}`);

  // End the run
  await tracker.endRun('FINISHED');
  console.log('\n✅ MLflow run completed successfully!');
  console.log(`   View results at: http://localhost:5000/#/experiments/${experimentId}`);
}

// Run the example
main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
