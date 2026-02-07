/**
 * Tests for Security Modules
 */

import { ConfusionMatrixDefense } from "../security/confusionMatrixDefense";

describe("ConfusionMatrixDefense", () => {
  it("should track confusion matrix metrics", () => {
    const defense = new ConfusionMatrixDefense({
      fpBudget: 0.1,
      fnBudget: 0.1,
    });

    // Add some predictions
    defense.update(1, 1, 0.9); // TP
    defense.update(0, 0, 0.8); // TN
    defense.update(1, 0, 0.7); // FP
    defense.update(0, 1, 0.6); // FN

    const metrics = defense.computeMetrics();

    expect(metrics.truePositives).toBe(1);
    expect(metrics.trueNegatives).toBe(1);
    expect(metrics.falsePositives).toBe(1);
    expect(metrics.falseNegatives).toBe(1);
    expect(metrics.precision).toBeGreaterThanOrEqual(0);
    expect(metrics.recall).toBeGreaterThanOrEqual(0);
  });
});
