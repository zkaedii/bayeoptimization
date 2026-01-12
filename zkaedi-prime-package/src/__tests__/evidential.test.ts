/**
 * Tests for Evidential Learning
 */

import { EvidentialClassifier } from "../evidential/evidentialClassification.js";

describe("EvidentialClassifier", () => {
  it("should classify with uncertainty", () => {
    const classifier = new EvidentialClassifier({
      nClasses: 5,
    });

    const evidence = [2.5, 0.3, 0.1, 1.2, 0.2];
    const output = classifier.forward(evidence);
    const prediction = classifier.predict(output);

    expect(prediction.predictedClass).toBeGreaterThanOrEqual(0);
    expect(prediction.predictedClass).toBeLessThan(5);
    expect(prediction.confidence).toBeGreaterThanOrEqual(0);
    expect(prediction.confidence).toBeLessThanOrEqual(1);
    expect(prediction.uncertainty).toBeGreaterThanOrEqual(0);
    expect(prediction.uncertainty).toBeLessThanOrEqual(1);
  });

  it("should detect high uncertainty", () => {
    const classifier = new EvidentialClassifier({
      nClasses: 5,
    });

    const lowEvidence = [0.1, 0.1, 0.1, 0.1, 0.1];
    const output = classifier.forward(lowEvidence);

    expect(classifier.isHighUncertainty(output, 0.5)).toBe(true);
  });
});
