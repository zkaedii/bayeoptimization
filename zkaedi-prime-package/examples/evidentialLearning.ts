/**
 * 🔱 ZKAEDI PRIME — Evidential Learning Example
 */

import { EvidentialClassifier } from "../src/evidential/evidentialClassification.js";

async function example() {
  // Create classifier for 10 classes
  const classifier = new EvidentialClassifier({
    nClasses: 10,
    klWeight: 0.001,
    klAnnealing: true,
  });

  // Example evidence vector (from neural network)
  const evidence = [2.5, 0.3, 0.1, 1.2, 0.2, 0.5, 0.8, 0.4, 0.6, 0.3];

  // Forward pass
  const output = classifier.forward(evidence);
  const prediction = classifier.predict(output);

  console.log("Evidential Classification Results:");
  console.log(`Predicted class: ${prediction.predictedClass}`);
  console.log(`Confidence: ${prediction.confidence.toFixed(4)}`);
  console.log(`Uncertainty: ${prediction.uncertainty.toFixed(4)}`);
  console.log(`Evidence: [${evidence.map((e) => e.toFixed(2)).join(", ")}]`);

  // Check if high uncertainty
  if (classifier.isHighUncertainty(output, 0.5)) {
    console.log("⚠️  High uncertainty - consider manual review");
  }
}

example().catch(console.error);
