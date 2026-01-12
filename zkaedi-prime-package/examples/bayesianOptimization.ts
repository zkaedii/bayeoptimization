/**
 * 🔱 ZKAEDI PRIME — Bayesian Optimization Example
 */

import { BayesianOptimizer } from "../src/optimization/bayesianOptimization.js";

async function example() {
  // Define search space
  const bounds = {
    x: [-5, 5],
    y: [-5, 5],
  };

  // Create optimizer
  const optimizer = new BayesianOptimizer(bounds, {
    nIter: 50,
    nWarmup: 10,
    verbose: true,
  });

  // Optimize (minimize negative = maximize)
  const result = await optimizer.optimize(async (params) => {
    const [x, y] = params;
    // Example: Rastrigin function
    const A = 10;
    const n = 2;
    const sum = x ** 2 - A * Math.cos(2 * Math.PI * x) + y ** 2 - A * Math.cos(2 * Math.PI * y);
    return A * n + sum;
  });

  console.log("Optimization Results:");
  console.log(`Best parameters: [${result.bestX.map((x) => x.toFixed(4)).join(", ")}]`);
  console.log(`Best value: ${result.bestY.toFixed(4)}`);
  console.log(`Iterations: ${result.nIter}`);
  console.log(`Trajectory length: ${result.trajectory.length}`);
}

example().catch(console.error);
