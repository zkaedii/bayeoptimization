/**
 * ZKAEDI PRIME - WebSocket Handler for Real-time Bayesian Optimization
 *
 * Streams optimization steps to connected clients over WebSocket.
 * Supports standard and PRIME (Hamiltonian) optimizer modes.
 */

import type { WebSocket, WebSocketServer } from "ws";
import { z } from "zod";
import pino from "pino";
import { TEST_FUNCTIONS } from "./test_functions";

const logger = pino({ name: "zkaedi-ws" });

/** Per-connection rate limiter: max messages per window. */
const WS_RATE_LIMIT_WINDOW_MS = 60_000;
const WS_RATE_LIMIT_MAX = 20;

/** Schema for optimization start messages from client. */
const StartMessageSchema = z.object({
  type: z.literal("START"),
  function: z
    .enum(["ackley", "rastrigin", "rosenbrock", "schwefel", "himmelblau"])
    .default("ackley"),
  optimizer: z.enum(["standard", "prime"]).default("standard"),
  n_iterations: z.number().int().min(1).max(500).default(50),
  bounds: z
    .array(z.tuple([z.number(), z.number()]))
    .optional(),
});

type StartMessage = z.infer<typeof StartMessageSchema>;

/** Step message sent to client during optimization. */
interface StepMessage {
  type: "STEP";
  step: number;
  candidate: number[];
  observed: number;
  best_so_far: number;
  acquisition_value: number;
  prime_state?: {
    phase: string;
    variance: number;
  };
}

/** Completion message sent to client. */
interface CompleteMessage {
  type: "COMPLETE";
  best_x: number[];
  best_y: number;
  total_steps: number;
  convergence_history: number[];
}

/** Error message sent to client. */
interface ErrorMessage {
  type: "ERROR";
  message: string;
}

/**
 * Simple in-memory GP surrogate for WebSocket-based BO demo.
 *
 * Uses random-restart local optimization of acquisition function
 * with a simplified GP model for real-time streaming performance.
 */
class SimpleGPOptimizer {
  private readonly bounds: [number, number][];
  private readonly nWarmup: number;
  private readonly usePrime: boolean;
  private readonly eta: number;
  private readonly gamma: number;

  private xObs: number[][] = [];
  private yObs: number[] = [];
  private bestX: number[] = [];
  private bestY: number = Infinity;

  /** Current Hamiltonian field state (PRIME mode only). */
  private hField: number = 0;
  private primeStep: number = 0;

  constructor(
    bounds: [number, number][],
    options: {
      nWarmup?: number;
      usePrime?: boolean;
      eta?: number;
      gamma?: number;
    } = {}
  ) {
    this.bounds = bounds;
    this.nWarmup = options.nWarmup ?? 5;
    this.usePrime = options.usePrime ?? false;
    this.eta = options.eta ?? 0.4;
    this.gamma = options.gamma ?? 0.3;
  }

  /**
   * Suggest next candidate point.
   *
   * During warmup, uses random sampling. After warmup, uses a
   * simplified acquisition function with optional PRIME modulation.
   *
   * @returns Candidate point and acquisition value
   *
   * @example
   * ```ts
   * const opt = new SimpleGPOptimizer([[-5, 5], [-5, 5]]);
   * const { candidate, acquisitionValue } = opt.suggest();
   * ```
   */
  suggest(): { candidate: number[]; acquisitionValue: number } {
    if (this.xObs.length < this.nWarmup) {
      const candidate = this.bounds.map(
        ([lo, hi]) => lo + Math.random() * (hi - lo)
      );
      return { candidate, acquisitionValue: 0 };
    }

    // Simple acquisition: sample candidates and pick best predicted improvement
    const nCandidates = 100;
    let bestCandidate = this.randomPoint();
    let bestAcq = -Infinity;

    for (let i = 0; i < nCandidates; i++) {
      const c = this.randomPoint();
      const acq = this.acquisitionEI(c);
      if (acq > bestAcq) {
        bestAcq = acq;
        bestCandidate = c;
      }
    }

    return { candidate: bestCandidate, acquisitionValue: bestAcq };
  }

  /**
   * Update optimizer with new observation.
   *
   * @param x - Observed point
   * @param y - Observed value (minimization)
   *
   * @example
   * ```ts
   * opt.update([1.0, 2.0], 3.5);
   * ```
   */
  update(x: number[], y: number): void {
    this.xObs.push([...x]);
    this.yObs.push(y);
    if (y < this.bestY) {
      this.bestY = y;
      this.bestX = [...x];
    }
    if (this.usePrime) {
      this.evolvePrime(y);
    }
  }

  /**
   * Get current PRIME field state.
   *
   * @returns Phase, variance, and step count
   *
   * @example
   * ```ts
   * const state = opt.getPrimeState();
   * console.log(state.phase); // "EXPLORING"
   * ```
   */
  getPrimeState(): { phase: string; variance: number } {
    const variance = Math.abs(this.hField);
    let phase: string;
    if (variance < 0.1) {
      phase = "CONVERGING";
    } else if (variance > 2.0) {
      phase = "BIFURCATING";
    } else {
      phase = "EXPLORING";
    }
    return { phase, variance };
  }

  /** Get best observed point and value. */
  getBest(): { bestX: number[]; bestY: number } {
    return { bestX: [...this.bestX], bestY: this.bestY };
  }

  /** Get convergence history (best-so-far at each step). */
  getConvergenceHistory(): number[] {
    const history: number[] = [];
    let best = Infinity;
    for (const y of this.yObs) {
      best = Math.min(best, y);
      history.push(best);
    }
    return history;
  }

  private randomPoint(): number[] {
    return this.bounds.map(([lo, hi]) => lo + Math.random() * (hi - lo));
  }

  /** Simplified Expected Improvement using nearest-neighbor GP approximation. */
  private acquisitionEI(x: number[]): number {
    if (this.xObs.length === 0) return 0;

    // Inverse-distance weighted prediction
    let wSum = 0;
    let mu = 0;
    let varAcc = 0;

    for (let i = 0; i < this.xObs.length; i++) {
      const dist = this.euclidean(x, this.xObs[i]);
      const w = 1 / (dist + 1e-6);
      wSum += w;
      mu += w * this.yObs[i];
    }
    mu /= wSum;

    for (let i = 0; i < this.xObs.length; i++) {
      const dist = this.euclidean(x, this.xObs[i]);
      const w = 1 / (dist + 1e-6);
      varAcc += (w / wSum) * (this.yObs[i] - mu) ** 2;
    }
    const sigma = Math.sqrt(varAcc + 1e-8);

    // EI for minimization
    const improvement = this.bestY - mu;
    const z = improvement / sigma;
    const ei = improvement * this.normalCDF(z) + sigma * this.normalPDF(z);

    // Apply PRIME modulation if enabled
    if (this.usePrime) {
      const sigmoid = 1 / (1 + Math.exp(-this.gamma * this.hField));
      return ei * (1 + this.eta * sigmoid);
    }

    return ei;
  }

  /** Evolve Hamiltonian field with Box-Muller noise. */
  private evolvePrime(y: number): void {
    const hBase = -y;
    const sigmoid = 1 / (1 + Math.exp(-this.gamma * this.hField));

    // Box-Muller noise
    const u1 = Math.random();
    const u2 = Math.random();
    const noise =
      Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);

    this.hField =
      hBase +
      this.eta * this.hField * sigmoid +
      0.05 * noise * (1 + 0.1 * Math.abs(this.hField));
    this.primeStep++;
  }

  private euclidean(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i] - b[i]) ** 2;
    }
    return Math.sqrt(sum);
  }

  private normalPDF(x: number): number {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  }

  private normalCDF(x: number): number {
    // Abramowitz and Stegun approximation
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804014327;
    const p =
      d *
      Math.exp(-0.5 * x * x) *
      (t *
        (0.3193815 +
          t * (-0.3565638 + t * (1.781478 + t * (-1.8212560 + t * 1.330274)))));
    return x > 0 ? 1 - p : p;
  }
}

/**
 * Run a full optimization session over WebSocket.
 *
 * Streams STEP messages for each iteration, then sends COMPLETE
 * when finished. Errors are caught and sent as ERROR messages.
 *
 * @param ws - WebSocket connection
 * @param config - Parsed start message
 *
 * @example
 * ```ts
 * // Called internally by setupWebSocket handler
 * await runOptimization(ws, { type: "START", function: "ackley", ... });
 * ```
 */
async function runOptimization(
  ws: WebSocket,
  config: StartMessage
): Promise<void> {
  const objectiveFn = TEST_FUNCTIONS[config.function];
  const bounds: [number, number][] = config.bounds ?? [
    [-5, 5],
    [-5, 5],
  ];

  const optimizer = new SimpleGPOptimizer(bounds, {
    nWarmup: Math.min(10, Math.floor(config.n_iterations * 0.2)),
    usePrime: config.optimizer === "prime",
  });

  for (let step = 0; step < config.n_iterations; step++) {
    if (ws.readyState !== 1) {
      logger.info("Client disconnected during optimization");
      return;
    }

    const { candidate, acquisitionValue } = optimizer.suggest();
    const observed = objectiveFn(candidate);
    optimizer.update(candidate, observed);

    const stepMsg: StepMessage = {
      type: "STEP",
      step: step + 1,
      candidate,
      observed,
      best_so_far: optimizer.getBest().bestY,
      acquisition_value: acquisitionValue,
    };

    if (config.optimizer === "prime") {
      stepMsg.prime_state = optimizer.getPrimeState();
    }

    ws.send(JSON.stringify(stepMsg));

    // Small delay for real-time streaming effect
    await new Promise((resolve) => setTimeout(resolve, 30));
  }

  const { bestX, bestY } = optimizer.getBest();
  const completeMsg: CompleteMessage = {
    type: "COMPLETE",
    best_x: bestX,
    best_y: bestY,
    total_steps: config.n_iterations,
    convergence_history: optimizer.getConvergenceHistory(),
  };

  ws.send(JSON.stringify(completeMsg));
  logger.info(
    { function: config.function, bestY, steps: config.n_iterations },
    "Optimization complete"
  );
}

/**
 * Set up WebSocket connection handling on the given server.
 *
 * Listens for START messages, validates with Zod, and runs
 * optimization sessions with real-time streaming.
 *
 * @param wss - WebSocket server instance
 *
 * @example
 * ```ts
 * import { WebSocketServer } from "ws";
 * const wss = new WebSocketServer({ server: httpServer });
 * setupWebSocket(wss);
 * ```
 */
export function setupWebSocket(wss: WebSocketServer): void {
  wss.on("connection", (ws: WebSocket) => {
    logger.info("Client connected");

    // Per-connection state
    let isRunning = false;
    const messageTimestamps: number[] = [];

    ws.on("message", async (data: Buffer | string) => {
      try {
        // Rate limiting: sliding window
        const now = Date.now();
        // Remove timestamps outside the window
        while (messageTimestamps.length > 0 && now - messageTimestamps[0] > WS_RATE_LIMIT_WINDOW_MS) {
          messageTimestamps.shift();
        }
        if (messageTimestamps.length >= WS_RATE_LIMIT_MAX) {
          const errorMsg: ErrorMessage = {
            type: "ERROR",
            message: "Rate limit exceeded. Max 20 messages per minute.",
          };
          ws.send(JSON.stringify(errorMsg));
          return;
        }
        messageTimestamps.push(now);

        const raw = JSON.parse(data.toString());
        const parsed = StartMessageSchema.safeParse(raw);

        if (!parsed.success) {
          const errorMsg: ErrorMessage = {
            type: "ERROR",
            message: `Invalid message: ${parsed.error.issues.map((i) => i.message).join(", ")}`,
          };
          ws.send(JSON.stringify(errorMsg));
          return;
        }

        // Guard against concurrent optimization runs on same socket
        if (isRunning) {
          const errorMsg: ErrorMessage = {
            type: "ERROR",
            message: "An optimization is already running on this connection.",
          };
          ws.send(JSON.stringify(errorMsg));
          return;
        }

        isRunning = true;
        try {
          await runOptimization(ws, parsed.data);
        } finally {
          isRunning = false;
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        logger.error({ err }, "WebSocket error");
        if (ws.readyState === 1) {
          const errorMsg: ErrorMessage = { type: "ERROR", message };
          ws.send(JSON.stringify(errorMsg));
        }
      }
    });

    ws.on("close", () => {
      logger.info("Client disconnected");
    });
  });
}
