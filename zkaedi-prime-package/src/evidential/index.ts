/**
 * 🔱 ZKAEDI PRIME — Evidential Learning Module
 * 
 * Evidential Deep Learning with uncertainty quantification
 */

export {
  EvidentialClassifier,
  type EvidentialClassifierOptions,
  type EvidentialOutput,
  type EvidentialPrediction,
} from "./evidentialClassification.js";

export {
  EvidentialRegressor,
  type EvidentialRegressorOptions,
  type EvidentialRegressionOutput,
} from "./evidentialRegression.js";

export {
  ZkaediPrimeOpenSetRecognition,
  type OpenSetRecognitionOptions,
  type OpenSetResult,
} from "./openSetRecognition.js";

export {
  ZkaediPrimeMultimodalFusion,
  type MultimodalFusionOptions,
  type MultimodalFusionResult,
  type ModalityEncoder,
} from "./multimodalFusion.js";
