/**
 * 🔱 ZKAEDI PRIME — Security Module
 * 
 * Confusion matrix defense and adversarial robustness
 */

export {
  ConfusionMatrixDefense,
  type ConfusionMatrixDefenseOptions,
  type ConfusionMatrixMetrics,
} from "./confusionMatrixDefense.js";

export {
  FalseNegativeHardening,
  type FNHardeningOptions,
  type FNHardeningResult,
} from "./falseNegativeHardening.js";

export {
  ZkaediPrimeAdversarialRobustness,
  type AdversarialRobustnessOptions,
  type AdversarialRobustnessResult,
} from "./adversarialRobustness.js";
