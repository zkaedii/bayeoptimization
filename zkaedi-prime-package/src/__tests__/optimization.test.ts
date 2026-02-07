/**
 * Tests for Bayesian Optimization
 */

import { BayesianOptimizer } from "../optimization/bayesianOptimization";

describe("BayesianOptimizer", () => {
  it("should optimize a simple function", async () => {
    const optimizer = new BayesianOptimizer(
      {
        x: [-5, 5],
        y: [-5, 5],
      },
      { nIter: 10, nWarmup: 5, verbose: false }
    );

    const result = await optimizer.optimize(async (params) => {
      const [x, y] = params;
      return x * x + y * y; // Minimize
    });

    expect(result.bestX).toHaveLength(2);
    expect(result.bestY).toBeLessThanOrEqual(100);
    expect(result.trajectory.length).toBeGreaterThan(0);
  });
});
