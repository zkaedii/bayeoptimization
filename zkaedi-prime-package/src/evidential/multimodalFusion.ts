/**
 * 🔱 ZKAEDI PRIME — Multimodal Fusion
 * 
 * Fuse heterogeneous data modalities via Hamiltonian consensus
 */

export interface ModalityEncoder {
  encode: (features: any) => number[];
  evidenceHead: (features: number[]) => number[];
  outputDim: number;
}

export interface MultimodalFusionOptions {
  modalities: Record<string, ModalityEncoder>;
  numClasses: number;
}

export interface MultimodalFusionResult {
  phase: "CONSENSUS" | "CONFLICT" | "UNCERTAIN";
  uncertainty: number;
  consensusUncertainty?: number;
  conflictUncertainty?: number;
}

/**
 * Multimodal Fusion via Hamiltonian Consensus
 */
export class ZkaediPrimeMultimodalFusion {
  private modalities: Record<string, ModalityEncoder>;
  private numClasses: number;

  constructor(options: MultimodalFusionOptions) {
    this.modalities = options.modalities;
    this.numClasses = options.numClasses;
  }

  /**
   * Forward pass: fuse modalities
   */
  forward(inputs: Record<string, { features: any }>): MultimodalFusionResult {
    const modalityNames = Object.keys(this.modalities);
    const evidenceVectors: number[][] = [];

    // Extract evidence from each modality
    for (const name of modalityNames) {
      const encoder = this.modalities[name];
      const input = inputs[name];
      if (!input) continue;

      const encoded = encoder.encode(input.features);
      const evidence = encoder.evidenceHead(encoded);
      evidenceVectors.push(evidence);
    }

    if (evidenceVectors.length === 0) {
      return {
        phase: "UNCERTAIN",
        uncertainty: 1.0,
      };
    }

    // Combine evidence (Product-of-Experts)
    const combinedEvidence = this.combineEvidence(evidenceVectors);
    const S = combinedEvidence.reduce((sum, e) => sum + e, 0) + this.numClasses;
    const uncertainty = this.numClasses / S;

    // Check consensus vs conflict
    const consensus = this.checkConsensus(evidenceVectors);
    const conflict = this.checkConflict(evidenceVectors);

    if (conflict) {
      return {
        phase: "CONFLICT",
        uncertainty,
        conflictUncertainty: uncertainty,
      };
    } else if (consensus) {
      return {
        phase: "CONSENSUS",
        uncertainty,
        consensusUncertainty: uncertainty,
      };
    } else {
      return {
        phase: "UNCERTAIN",
        uncertainty,
      };
    }
  }

  private combineEvidence(evidenceVectors: number[][]): number[] {
    // Product-of-Experts: multiply evidence
    const combined = Array(this.numClasses).fill(1.0);
    for (const evidence of evidenceVectors) {
      for (let i = 0; i < this.numClasses; i++) {
        combined[i] *= Math.max(0.1, evidence[i] || 0.1);
      }
    }
    return combined;
  }

  private checkConsensus(evidenceVectors: number[][]): boolean {
    // Check if modalities agree on top class
    const topClasses = evidenceVectors.map((ev) => {
      return ev.indexOf(Math.max(...ev));
    });
    const allSame = topClasses.every((c) => c === topClasses[0]);
    return allSame;
  }

  private checkConflict(evidenceVectors: number[][]): boolean {
    // Check if modalities strongly disagree
    const topClasses = evidenceVectors.map((ev) => {
      return ev.indexOf(Math.max(...ev));
    });
    const uniqueClasses = new Set(topClasses);
    return uniqueClasses.size > 1 && evidenceVectors.length > 1;
  }
}
