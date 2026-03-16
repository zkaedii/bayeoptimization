/**
 * ZKAEDI PRIME - Express Server
 *
 * Production-grade REST API with 5 endpoint groups plus WebSocket
 * real-time Bayesian optimization streaming. Includes Zod validation,
 * rate limiting, security headers, and structured logging.
 */

import express, { type Request, type Response, type NextFunction } from "express";
import http from "http";
import { WebSocketServer } from "ws";
import { z, ZodError } from "zod";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import pino from "pino";
import pinoHttp from "pino-http";
import path from "path";
import { setupWebSocket } from "./ws_handler";
import { TEST_FUNCTIONS } from "./test_functions";

// ═══════════════════════════════════════════════
//  CONFIGURATION
// ═══════════════════════════════════════════════

const PORT = parseInt(process.env["PORT"] ?? "3000", 10);
const ALLOWED_ORIGINS = (process.env["ALLOWED_ORIGINS"] ?? "").split(",").filter(Boolean);

const logger = pino({ name: "zkaedi-server" });

// ═══════════════════════════════════════════════
//  REQUEST SCHEMAS (Zod)
// ═══════════════════════════════════════════════

const ClassifySchema = z.object({
  x: z.array(z.number()),
  model_id: z.string().optional(),
});

const RegressSchema = z.object({
  x: z.array(z.number()),
});

const OpenSetSchema = z.object({
  x: z.array(z.number()),
  threshold: z.number().min(0).max(1).optional(),
});

const AdversarialEvaluateSchema = z.object({
  x: z.array(z.number()),
  y: z.number(),
  epsilon_list: z.array(z.number().positive()).optional(),
});

const AdversarialAttackSchema = z.object({
  x: z.array(z.number()),
  y: z.number(),
  epsilon: z.number().positive().default(0.1),
});

const ActiveInitSchema = z.object({
  pool: z.array(z.array(z.number())),
  w_uncertainty: z.number().min(0).max(1).optional(),
  w_diversity: z.number().min(0).max(1).optional(),
});

const ActiveStepSchema = z.object({
  session_id: z.string(),
  k: z.number().int().min(1).max(100),
});

const DriftScenarioSchema = z.object({
  reference: z.array(z.array(z.number())),
  current: z.array(z.array(z.number())),
  window_size: z.number().int().positive().optional(),
});

// ═══════════════════════════════════════════════
//  SIMULATION ENGINES
// ═══════════════════════════════════════════════

/**
 * Simulate evidential classification with Dirichlet uncertainty.
 *
 * @param x - Feature vector
 * @param nClasses - Number of classes
 * @returns Classification result with uncertainty decomposition
 *
 * @example
 * ```ts
 * simulateClassify([0.1, 0.2, 0.3], 5);
 * ```
 */
function simulateClassify(x: number[], nClasses: number = 5): Record<string, unknown> {
  const evidence = Array.from({ length: nClasses }, (_, i) => {
    let e = 0;
    for (const xi of x) {
      e += Math.abs(Math.sin(xi * (i + 1) * 2.1));
    }
    return Math.max(0, e);
  });

  const alpha = evidence.map((e) => e + 1);
  const S = alpha.reduce((a, b) => a + b, 0);
  const probs = alpha.map((a) => a / S);
  const uncertainty = nClasses / S;

  const maxIdx = probs.indexOf(Math.max(...probs));
  const aleatoric = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
  const epistemic = Math.max(0, uncertainty - aleatoric / Math.log(nClasses));

  return {
    class: maxIdx,
    probs,
    uncertainty,
    epistemic,
    aleatoric,
    phase: uncertainty > 0.7 ? "UNCERTAIN" : uncertainty > 0.3 ? "EXPLORING" : "CONFIDENT",
  };
}

/**
 * Simulate evidential regression with NIG uncertainty.
 *
 * @param x - Feature vector
 * @returns Regression result with uncertainty split
 *
 * @example
 * ```ts
 * simulateRegress([1.0, 2.0]);
 * ```
 */
function simulateRegress(x: number[]): Record<string, number> {
  const mean = x.reduce((a, b) => a + Math.sin(b), 0) / x.length;
  const nu = 5 + x.length;
  const alpha = 3.0;
  const beta = 0.5 + Math.abs(mean) * 0.3;

  const aleatoric = beta / (alpha - 1);
  const epistemic = beta / (nu * (alpha - 1));

  return { mean, aleatoric, epistemic, total: aleatoric + epistemic };
}

/**
 * Simulate open-set recognition with rejection.
 *
 * @param x - Feature vector
 * @param threshold - Rejection threshold (vacuity above → unknown)
 * @returns Classification with possible rejection
 *
 * @example
 * ```ts
 * simulateOpenSet([0.1, 0.2], 0.5);
 * ```
 */
function simulateOpenSet(
  x: number[],
  threshold: number = 0.5
): Record<string, unknown> {
  const result = simulateClassify(x, 5);
  const uncertainty = result["uncertainty"] as number;
  const rejected = uncertainty > threshold;
  return {
    ...result,
    class: rejected ? "UNKNOWN" : result["class"],
    rejected,
  };
}

/**
 * Simulate FGSM adversarial attack on input.
 *
 * @param x - Original input
 * @param epsilon - Perturbation magnitude
 * @returns Adversarial example and metadata
 *
 * @example
 * ```ts
 * simulateAttack([0.5, 0.3], 0.1);
 * ```
 */
function simulateAttack(
  x: number[],
  epsilon: number
): Record<string, unknown> {
  // Simulate gradient as sign-alternating based on position
  const gradient = x.map((xi, i) => Math.cos(xi * (i + 1)));
  const xAdv = x.map((xi, i) => {
    const sign = gradient[i] >= 0 ? 1 : -1;
    return Math.max(0, Math.min(1, xi + epsilon * sign));
  });

  const pertNorm = Math.sqrt(
    x.reduce((sum, xi, i) => sum + (xi - xAdv[i]) ** 2, 0)
  );

  return {
    x_adv: xAdv,
    perturbation_norm: pertNorm,
    success: pertNorm > epsilon * 0.5,
  };
}

/**
 * Simulate adversarial robustness evaluation across epsilon sweep.
 *
 * @param x - Clean input
 * @param y - True label
 * @param epsilonList - Epsilons to evaluate
 * @returns Full robustness report
 *
 * @example
 * ```ts
 * simulateEvaluation([0.5], 1, [0.01, 0.1]);
 * ```
 */
function simulateEvaluation(
  x: number[],
  y: number,
  epsilonList: number[]
): Record<string, unknown> {
  const cleanResult = simulateClassify(x);
  const cleanCorrect = (cleanResult["class"] as number) === y ? 1 : 0;

  const adversarialResults: Record<string, number> = {};
  const defenseResults: Record<string, number> = {};
  const perSample: Record<string, unknown>[] = [];

  for (const eps of epsilonList) {
    const attack = simulateAttack(x, eps);
    const advResult = simulateClassify(attack["x_adv"] as number[]);
    const advCorrect = (advResult["class"] as number) === y ? 1 : 0;

    // Hamiltonian defense: smooth perturbation
    const defended = (attack["x_adv"] as number[]).map(
      (xi, i) => xi * 0.7 + x[i] * 0.3
    );
    const defResult = simulateClassify(defended);
    const defCorrect = (defResult["class"] as number) === y ? 1 : 0;

    adversarialResults[eps.toString()] = advCorrect;
    defenseResults[eps.toString()] = defCorrect;
    perSample.push({ epsilon: eps, advCorrect, defCorrect });
  }

  return {
    clean_accuracy: cleanCorrect,
    adversarial_results: adversarialResults,
    defense_results: defenseResults,
    per_sample: perSample,
  };
}

// Active learning session storage
interface ActiveSession {
  pool: number[][];
  labeled: Set<number>;
  wUncertainty: number;
  wDiversity: number;
  createdAt: number;
}

const MAX_SESSIONS = 1000;
const SESSION_TTL_MS = 30 * 60 * 1000; // 30 minutes

const activeSessions = new Map<string, ActiveSession>();

/**
 * Evict expired sessions and enforce max session cap (LRU by creation time).
 *
 * @example
 * ```ts
 * evictStaleSessions();
 * ```
 */
function evictStaleSessions(): void {
  const now = Date.now();
  for (const [id, session] of activeSessions) {
    if (now - session.createdAt > SESSION_TTL_MS) {
      activeSessions.delete(id);
    }
  }
  // If still over cap, evict oldest sessions
  if (activeSessions.size > MAX_SESSIONS) {
    const entries = Array.from(activeSessions.entries())
      .sort((a, b) => a[1].createdAt - b[1].createdAt);
    const toEvict = entries.slice(0, activeSessions.size - MAX_SESSIONS);
    for (const [id] of toEvict) {
      activeSessions.delete(id);
    }
  }
}

// ═══════════════════════════════════════════════
//  EXPRESS APP
// ═══════════════════════════════════════════════

const app = express();

// Security headers
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        connectSrc: ["'self'", "ws:", "wss:"],
      },
    },
  })
);

// Rate limiting: 100 req/min per IP
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Rate limit exceeded. Try again in 1 minute." },
});
app.use("/api/", limiter);

// Structured logging
app.use(pinoHttp({ logger }));

// Body parsing
app.use(express.json({ limit: "1mb" }));

// CORS
app.use((req: Request, res: Response, next: NextFunction) => {
  const origin = req.headers.origin;
  if (ALLOWED_ORIGINS.length === 0 || (origin && ALLOWED_ORIGINS.includes(origin))) {
    res.header("Access-Control-Allow-Origin", origin ?? "*");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.header("Access-Control-Allow-Headers", "Content-Type");
  }
  if (req.method === "OPTIONS") {
    res.sendStatus(204);
    return;
  }
  next();
});

// Static files
app.use(express.static(path.join(__dirname, "../../app/public")));

// ═══════════════════════════════════════════════
//  VALIDATION MIDDLEWARE
// ═══════════════════════════════════════════════

/**
 * Create an Express middleware that validates request body with a Zod schema.
 *
 * @param schema - Zod schema to validate against
 * @returns Express middleware
 *
 * @example
 * ```ts
 * app.post("/api/classify", validate(ClassifySchema), handler);
 * ```
 */
function validate(schema: z.ZodType) {
  return (req: Request, res: Response, next: NextFunction): void => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      res.status(400).json({
        error: "Validation failed",
        issues: result.error.issues.map((i) => ({
          path: i.path.join("."),
          message: i.message,
        })),
      });
      return;
    }
    req.body = result.data;
    next();
  };
}

// ═══════════════════════════════════════════════
//  REST API ROUTES
// ═══════════════════════════════════════════════

app.post("/api/classify", validate(ClassifySchema), (req: Request, res: Response) => {
  const { x, model_id } = req.body as z.infer<typeof ClassifySchema>;
  const result = simulateClassify(x);
  res.json({ ...result, model_id: model_id ?? "default" });
});

app.post("/api/regress", validate(RegressSchema), (req: Request, res: Response) => {
  const { x } = req.body as z.infer<typeof RegressSchema>;
  res.json(simulateRegress(x));
});

app.post("/api/openset", validate(OpenSetSchema), (req: Request, res: Response) => {
  const { x, threshold } = req.body as z.infer<typeof OpenSetSchema>;
  res.json(simulateOpenSet(x, threshold));
});

app.post(
  "/api/adversarial/evaluate",
  validate(AdversarialEvaluateSchema),
  (req: Request, res: Response) => {
    const { x, y, epsilon_list } = req.body as z.infer<typeof AdversarialEvaluateSchema>;
    const epsilons = epsilon_list ?? [0.01, 0.05, 0.1, 0.2, 0.3];
    res.json(simulateEvaluation(x, y, epsilons));
  }
);

app.post(
  "/api/adversarial/attack",
  validate(AdversarialAttackSchema),
  (req: Request, res: Response) => {
    const { x, epsilon } = req.body as z.infer<typeof AdversarialAttackSchema>;
    res.json(simulateAttack(x, epsilon));
  }
);

app.post("/api/active/init", validate(ActiveInitSchema), (req: Request, res: Response) => {
  const { pool, w_uncertainty, w_diversity } = req.body as z.infer<typeof ActiveInitSchema>;
  evictStaleSessions();
  const sessionId = `al_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  activeSessions.set(sessionId, {
    pool,
    labeled: new Set(),
    wUncertainty: w_uncertainty ?? 0.6,
    wDiversity: w_diversity ?? 0.4,
    createdAt: Date.now(),
  });
  res.json({ session_id: sessionId, pool_size: pool.length });
});

app.post("/api/active/step", validate(ActiveStepSchema), (req: Request, res: Response) => {
  const { session_id, k } = req.body as z.infer<typeof ActiveStepSchema>;
  const session = activeSessions.get(session_id);

  if (!session) {
    res.status(404).json({ error: "Session not found" });
    return;
  }

  const unlabeled = session.pool
    .map((_, i) => i)
    .filter((i) => !session.labeled.has(i));

  if (unlabeled.length === 0) {
    res.json({ selected_indices: [], scores: [], uncertainties: [], done: true });
    return;
  }

  // Uncertainty + diversity scoring
  const actualK = Math.min(k, unlabeled.length);
  const selected: number[] = [];
  const scores: number[] = [];
  const uncertainties: number[] = [];

  for (let pick = 0; pick < actualK; pick++) {
    let bestIdx = -1;
    let bestScore = -Infinity;
    let bestUncertainty = 0;

    for (const idx of unlabeled) {
      if (selected.includes(idx)) continue;

      // Uncertainty: variance-like score
      const point = session.pool[idx];
      const unc = point.reduce((s, v) => s + Math.abs(Math.sin(v * 3.7)), 0) / point.length;

      // Diversity: min distance to selected
      let minDist = Infinity;
      for (const selIdx of [...Array.from(session.labeled), ...selected]) {
        const selPoint = session.pool[selIdx];
        if (selPoint) {
          const dist = Math.sqrt(
            point.reduce((s, v, d) => s + (v - selPoint[d]) ** 2, 0)
          );
          minDist = Math.min(minDist, dist);
        }
      }
      if (minDist === Infinity) minDist = 1;

      const score = session.wUncertainty * unc + session.wDiversity * minDist;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
        bestUncertainty = unc;
      }
    }

    if (bestIdx >= 0) {
      selected.push(bestIdx);
      scores.push(bestScore);
      uncertainties.push(bestUncertainty);
      session.labeled.add(bestIdx);
    }
  }

  res.json({ selected_indices: selected, scores, uncertainties, done: false });
});

app.post(
  "/api/drift/scenario",
  validate(DriftScenarioSchema),
  (req: Request, res: Response) => {
    const { reference, current, window_size } = req.body as z.infer<typeof DriftScenarioSchema>;
    const ws = window_size ?? Math.min(reference.length, 50);

    // KL divergence on binned distributions
    const nBins = 10;
    const nDims = reference[0]?.length ?? 1;

    function binDistribution(data: number[][], dim: number, globalMin: number, globalMax: number): number[] {
      const values = data.map((d) => d[dim] ?? 0);
      const range = globalMax - globalMin || 1;
      const bins = new Array(nBins).fill(0);
      for (const v of values) {
        const idx = Math.min(nBins - 1, Math.max(0, Math.floor(((v - globalMin) / range) * nBins)));
        bins[idx]++;
      }
      const total = values.length;
      return bins.map((b) => (b + 1e-10) / (total + nBins * 1e-10));
    }

    let klTotal = 0;
    for (let d = 0; d < nDims; d++) {
      // Compute global min/max across both datasets for consistent binning
      const allVals = [...reference.map((r) => r[d] ?? 0), ...current.map((c) => c[d] ?? 0)];
      const globalMin = Math.min(...allVals);
      const globalMax = Math.max(...allVals);
      const refDist = binDistribution(reference, d, globalMin, globalMax);
      const curDist = binDistribution(current, d, globalMin, globalMax);
      for (let b = 0; b < nBins; b++) {
        klTotal += curDist[b] * Math.log(curDist[b] / refDist[b]);
      }
    }
    const klScore = klTotal / nDims;

    let phase: string;
    if (klScore < 0.1) {
      phase = "STABLE";
    } else if (klScore < 0.3) {
      phase = "DRIFT";
    } else {
      phase = "CRITICAL";
    }

    res.json({
      phase,
      kl_score: klScore,
      samples_seen: reference.length + current.length,
      window_size: ws,
      retrain_signal: phase === "CRITICAL",
    });
  }
);

// System info
app.get("/api/info", (_req: Request, res: Response) => {
  res.json({
    name: "ZKAEDI PRIME Interactive Lab",
    version: "2.0.0",
    modules: [
      "Bayesian Optimization",
      "Evidential Learning",
      "Adversarial Robustness",
      "Active Learning",
      "Drift Detection",
    ],
    objectives: Object.keys(TEST_FUNCTIONS),
  });
});

// ═══════════════════════════════════════════════
//  ERROR HANDLING
// ═══════════════════════════════════════════════

app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  if (err instanceof ZodError) {
    res.status(400).json({ error: "Validation error", issues: err.issues });
    return;
  }
  logger.error({ err }, "Unhandled error");
  res.status(500).json({ error: "Internal server error" });
});

// ═══════════════════════════════════════════════
//  SERVER STARTUP
// ═══════════════════════════════════════════════

const server = http.createServer(app);
const wss = new WebSocketServer({ server });
setupWebSocket(wss);

/**
 * Start the ZKAEDI PRIME server.
 *
 * Binds to PORT (default 3000) and sets up graceful shutdown
 * that drains WebSocket connections before exit.
 *
 * @example
 * ```ts
 * // Run directly: npx ts-node src/server/app.ts
 * // Or via npm: npm run start:server
 * ```
 */
function startServer(): void {
  server.listen(PORT, () => {
    logger.info({ port: PORT }, "ZKAEDI PRIME server started");
    console.log(`
  ╔══════════════════════════════════════════════════╗
  ║                                                  ║
  ║     ⚡ ZKAEDI PRIME Interactive Lab v2.0 ⚡       ║
  ║                                                  ║
  ║     Server:    http://localhost:${PORT}              ║
  ║     WebSocket: ws://localhost:${PORT}               ║
  ║                                                  ║
  ║     Endpoints:                                   ║
  ║       POST /api/classify                         ║
  ║       POST /api/regress                          ║
  ║       POST /api/openset                          ║
  ║       POST /api/adversarial/evaluate             ║
  ║       POST /api/adversarial/attack               ║
  ║       POST /api/active/init                      ║
  ║       POST /api/active/step                      ║
  ║       POST /api/drift/scenario                   ║
  ║       GET  /api/info                             ║
  ║       WS   /ws/optimize                          ║
  ║                                                  ║
  ║     Rate limit: 100 req/min per IP               ║
  ║                                                  ║
  ╚══════════════════════════════════════════════════╝
  `);
  });

  // Graceful shutdown
  const shutdown = (): void => {
    logger.info("Shutting down...");
    wss.clients.forEach((client) => {
      client.close(1001, "Server shutting down");
    });
    server.close(() => {
      logger.info("Server closed");
      process.exit(0);
    });
    // Force close after 5s
    setTimeout(() => process.exit(1), 5000);
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

// Start if run directly
if (require.main === module) {
  startServer();
}

export { app, server, wss, startServer };
