/**
 * ZKAEDI PRIME Interactive Lab - Server
 * Express + WebSocket server powering the real-time ML platform
 */

const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

const { BayesianOptimizer } = require('./src/engine/bayesianOptimizer');
const { EvidentialClassifier, EvidentialRegressor, OpenSetRecognizer } = require('./src/engine/evidentialLearning');
const { AdversarialLab } = require('./src/engine/adversarialLab');
const { ActiveLearningSimulator } = require('./src/engine/activeLearning');
const { DriftDetector } = require('./src/engine/driftDetector');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ═══════════════════════════════════════════════
//  OBJECTIVE FUNCTIONS (for Bayesian Optimization)
// ═══════════════════════════════════════════════

const OBJECTIVES = {
  ackley: (params) => {
    const [x, y] = params;
    const a = 20, b = 0.2, c = 2 * Math.PI;
    const term1 = -a * Math.exp(-b * Math.sqrt(0.5 * (x * x + y * y)));
    const term2 = -Math.exp(0.5 * (Math.cos(c * x) + Math.cos(c * y)));
    return -(term1 + term2 + a + Math.E); // negate for maximization
  },
  rastrigin: (params) => {
    const A = 10;
    const n = params.length;
    let sum = A * n;
    for (const x of params) {
      sum += x * x - A * Math.cos(2 * Math.PI * x);
    }
    return -sum;
  },
  rosenbrock: (params) => {
    let sum = 0;
    for (let i = 0; i < params.length - 1; i++) {
      sum += 100 * (params[i + 1] - params[i] ** 2) ** 2 + (1 - params[i]) ** 2;
    }
    return -sum;
  },
  schwefel: (params) => {
    let sum = 418.9829 * params.length;
    for (const x of params) {
      sum -= x * Math.sin(Math.sqrt(Math.abs(x)));
    }
    return -sum;
  },
  himmelblau: (params) => {
    const [x, y] = params;
    return -((x * x + y - 11) ** 2 + (x + y * y - 7) ** 2);
  }
};

// ═══════════════════════════════════════════════
//  REST API ROUTES
// ═══════════════════════════════════════════════

// --- Evidential Learning ---
app.post('/api/evidential/classify', (req, res) => {
  const { nClasses = 5, batchSize = 20 } = req.body || {};
  const classifier = new EvidentialClassifier(nClasses);
  const results = classifier.generateSyntheticBatch(batchSize);
  res.json({ results });
});

app.post('/api/evidential/regression', (req, res) => {
  const { nPoints = 50 } = req.body || {};
  const regressor = new EvidentialRegressor();
  const results = regressor.generateSyntheticSeries(nPoints);
  res.json({ results });
});

app.post('/api/evidential/openset', (req, res) => {
  const { nClasses = 5, nKnown = 15, nUnknown = 5, threshold = 0.6 } = req.body || {};
  const recognizer = new OpenSetRecognizer(nClasses, threshold);
  const results = recognizer.generateMixedBatch(nKnown, nUnknown);
  res.json({ results });
});

// --- Adversarial Lab ---
app.post('/api/adversarial/evaluate', (req, res) => {
  const { epsilon = 0.1, nSamples = 30, smoothingStrength = 0.3 } = req.body || {};
  const lab = new AdversarialLab({ epsilon, smoothingStrength });
  const results = lab.evaluateRobustness(nSamples);
  res.json(results);
});

app.post('/api/adversarial/attack', (req, res) => {
  const { epsilon = 0.1, inputDim = 8 } = req.body || {};
  const lab = new AdversarialLab({ epsilon });
  const input = Array.from({ length: inputDim }, () => (Math.random() - 0.5) * 4);
  const attack = lab.fgsm(input);
  const defense = lab.hamiltonianSmoothing(attack.adversarialInput);
  res.json({ attack, defense });
});

// --- Active Learning ---
let activeLearningSim = null;

app.post('/api/active-learning/init', (req, res) => {
  const { nClasses = 3, nSamples = 200, uncertaintyWeight = 0.6, diversityWeight = 0.4 } = req.body || {};
  activeLearningSim = new ActiveLearningSimulator({ nClasses, uncertaintyWeight, diversityWeight });
  const state = activeLearningSim.generateDataset(nSamples);
  res.json(state);
});

app.post('/api/active-learning/step', (req, res) => {
  if (!activeLearningSim) {
    activeLearningSim = new ActiveLearningSimulator();
    activeLearningSim.generateDataset();
  }
  const { batchSize = 5 } = req.body || {};
  const result = activeLearningSim.selectBatch(batchSize);
  if (!result) {
    return res.json({ done: true, state: activeLearningSim.getState() });
  }
  res.json({ done: false, ...result, state: activeLearningSim.getState() });
});

// --- Drift Detection ---
app.post('/api/drift/scenario', (req, res) => {
  const { nTicks = 150, windowSize = 20, driftThreshold = 0.3, criticalThreshold = 0.6 } = req.body || {};
  const detector = new DriftDetector({ windowSize, driftThreshold, criticalThreshold });
  const results = detector.generateScenario(nTicks);
  res.json({ results, state: detector.getState() });
});

// --- System Info ---
app.get('/api/info', (req, res) => {
  res.json({
    name: 'ZKAEDI PRIME Interactive Lab',
    version: '1.0.0',
    modules: [
      'Bayesian Optimization',
      'Evidential Learning',
      'Adversarial Robustness',
      'Active Learning',
      'Drift Detection'
    ],
    objectives: Object.keys(OBJECTIVES)
  });
});

// ═══════════════════════════════════════════════
//  WEBSOCKET - Real-time Bayesian Optimization
// ═══════════════════════════════════════════════

wss.on('connection', (ws) => {
  console.log('[WS] Client connected');

  ws.on('message', async (data) => {
    try {
      const msg = JSON.parse(data);

      if (msg.type === 'start_optimization') {
        const {
          objective = 'ackley',
          bounds = [[-5, 5], [-5, 5]],
          nIter = 40,
          nWarmup = 5,
          acquisitionFunction = 'EI',
          kernel = 'rbf',
          eta = 0.1,
          gamma = 0.95
        } = msg.config || {};

        const objectiveFn = OBJECTIVES[objective] || OBJECTIVES.ackley;

        const optimizer = new BayesianOptimizer(bounds, {
          nIter,
          nWarmup,
          acquisitionFunction,
          kernel,
          eta,
          gamma
        });

        ws.send(JSON.stringify({
          type: 'optimization_started',
          config: { objective, bounds, nIter, nWarmup, acquisitionFunction, kernel }
        }));

        await optimizer.optimize(
          async (params) => objectiveFn(params),
          (step) => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                type: 'optimization_step',
                step
              }));
            }
          }
        );

        // Send final surface data
        const surface = optimizer.getSurface(25);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'optimization_complete',
            bestX: optimizer.bestX,
            bestY: optimizer.bestY,
            surface,
            totalIterations: optimizer.trajectory.length
          }));
        }
      }
    } catch (err) {
      console.error('[WS] Error:', err.message);
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'error', message: err.message }));
      }
    }
  });

  ws.on('close', () => {
    console.log('[WS] Client disconnected');
  });
});

// ═══════════════════════════════════════════════
//  START SERVER
// ═══════════════════════════════════════════════

server.listen(PORT, () => {
  console.log(`
  ╔══════════════════════════════════════════════════╗
  ║                                                  ║
  ║     ⚡ ZKAEDI PRIME Interactive Lab ⚡            ║
  ║                                                  ║
  ║     Server running on http://localhost:${PORT}      ║
  ║                                                  ║
  ║     Modules:                                     ║
  ║       • Bayesian Optimization (WebSocket)        ║
  ║       • Evidential Learning                      ║
  ║       • Adversarial Robustness                   ║
  ║       • Active Learning                          ║
  ║       • Drift Detection                          ║
  ║                                                  ║
  ╚══════════════════════════════════════════════════╝
  `);
});

module.exports = { app, server };
