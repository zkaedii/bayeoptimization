/**
 * ZKAEDI PRIME - Server Test Suite
 *
 * Tests all REST endpoints with valid + invalid inputs,
 * WebSocket optimization flow, rate limiting, and Zod validation.
 */

import request from "supertest";
import { app, server, wss } from "../src/server/app";
import { ackley, rastrigin, rosenbrock, schwefel, himmelblau, TEST_FUNCTIONS } from "../src/server/test_functions";
import WebSocket from "ws";

const PORT_TEST = 9876;

let httpServer: ReturnType<typeof server.listen>;

beforeAll((done) => {
  httpServer = server.listen(PORT_TEST, done);
});

afterAll((done) => {
  wss.clients.forEach((client) => client.close());
  httpServer.close(done);
});

// ═══════════════════════════════════════════════
//  TEST FUNCTIONS
// ═══════════════════════════════════════════════

describe("Test Functions", () => {
  test("ackley global minimum at origin", () => {
    const val = ackley([0, 0]);
    expect(val).toBeCloseTo(0, 10);
  });

  test("ackley with non-zero input", () => {
    expect(ackley([1, 1])).toBeGreaterThan(0);
  });

  test("ackley multidimensional", () => {
    expect(ackley([0, 0, 0, 0])).toBeCloseTo(0, 10);
  });

  test("rastrigin global minimum at origin", () => {
    expect(rastrigin([0, 0])).toBe(0);
  });

  test("rastrigin non-zero", () => {
    expect(rastrigin([1, 1])).toBeGreaterThan(0);
  });

  test("rosenbrock global minimum at (1,1,...)", () => {
    expect(rosenbrock([1, 1])).toBe(0);
    expect(rosenbrock([1, 1, 1])).toBe(0);
  });

  test("rosenbrock non-minimum", () => {
    expect(rosenbrock([0, 0])).toBeGreaterThan(0);
  });

  test("schwefel near global minimum", () => {
    const val = schwefel([420.9687, 420.9687]);
    expect(val).toBeCloseTo(0, 0);
  });

  test("himmelblau global minimum at (3,2)", () => {
    expect(himmelblau([3, 2])).toBe(0);
  });

  test("himmelblau other minima", () => {
    expect(himmelblau([-2.805118, 3.131312])).toBeCloseTo(0, 2);
  });

  test("TEST_FUNCTIONS map contains all functions", () => {
    expect(Object.keys(TEST_FUNCTIONS)).toEqual([
      "ackley",
      "rastrigin",
      "rosenbrock",
      "schwefel",
      "himmelblau",
    ]);
  });
});

// ═══════════════════════════════════════════════
//  REST API TESTS
// ═══════════════════════════════════════════════

describe("GET /api/info", () => {
  test("returns server info", async () => {
    const res = await request(app).get("/api/info");
    expect(res.status).toBe(200);
    expect(res.body.name).toBe("ZKAEDI PRIME Interactive Lab");
    expect(res.body.version).toBe("2.0.0");
    expect(res.body.modules).toHaveLength(5);
    expect(res.body.objectives).toEqual([
      "ackley",
      "rastrigin",
      "rosenbrock",
      "schwefel",
      "himmelblau",
    ]);
  });
});

describe("POST /api/classify", () => {
  test("valid classification", async () => {
    const res = await request(app)
      .post("/api/classify")
      .send({ x: [0.1, 0.2, 0.3] });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("class");
    expect(res.body).toHaveProperty("probs");
    expect(res.body).toHaveProperty("uncertainty");
    expect(res.body).toHaveProperty("epistemic");
    expect(res.body).toHaveProperty("aleatoric");
    expect(res.body).toHaveProperty("phase");
    expect(res.body.probs).toBeInstanceOf(Array);
    expect(typeof res.body.class).toBe("number");
  });

  test("with model_id", async () => {
    const res = await request(app)
      .post("/api/classify")
      .send({ x: [0.5], model_id: "test-model" });
    expect(res.status).toBe(200);
    expect(res.body.model_id).toBe("test-model");
  });

  test("rejects missing x", async () => {
    const res = await request(app).post("/api/classify").send({});
    expect(res.status).toBe(400);
    expect(res.body).toHaveProperty("error");
  });

  test("rejects invalid x type", async () => {
    const res = await request(app)
      .post("/api/classify")
      .send({ x: "not an array" });
    expect(res.status).toBe(400);
  });
});

describe("POST /api/regress", () => {
  test("valid regression", async () => {
    const res = await request(app)
      .post("/api/regress")
      .send({ x: [1.0, 2.0] });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("mean");
    expect(res.body).toHaveProperty("aleatoric");
    expect(res.body).toHaveProperty("epistemic");
    expect(res.body).toHaveProperty("total");
    expect(res.body.total).toBeCloseTo(
      res.body.aleatoric + res.body.epistemic,
      10
    );
  });

  test("rejects missing body", async () => {
    const res = await request(app).post("/api/regress").send({});
    expect(res.status).toBe(400);
  });
});

describe("POST /api/openset", () => {
  test("valid open-set classification", async () => {
    const res = await request(app)
      .post("/api/openset")
      .send({ x: [0.1, 0.2] });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("rejected");
    expect(res.body).toHaveProperty("uncertainty");
  });

  test("with custom threshold", async () => {
    const res = await request(app)
      .post("/api/openset")
      .send({ x: [0.1], threshold: 0.01 });
    expect(res.status).toBe(200);
    // Very low threshold → likely rejected
    expect(res.body.rejected).toBe(true);
    expect(res.body.class).toBe("UNKNOWN");
  });

  test("rejects invalid threshold", async () => {
    const res = await request(app)
      .post("/api/openset")
      .send({ x: [0.1], threshold: 2.0 });
    expect(res.status).toBe(400);
  });
});

describe("POST /api/adversarial/evaluate", () => {
  test("valid evaluation", async () => {
    const res = await request(app)
      .post("/api/adversarial/evaluate")
      .send({ x: [0.5, 0.3, 0.8], y: 0 });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("clean_accuracy");
    expect(res.body).toHaveProperty("adversarial_results");
    expect(res.body).toHaveProperty("defense_results");
    expect(res.body).toHaveProperty("per_sample");
  });

  test("with custom epsilon list", async () => {
    const res = await request(app)
      .post("/api/adversarial/evaluate")
      .send({ x: [0.5], y: 1, epsilon_list: [0.05, 0.1] });
    expect(res.status).toBe(200);
    expect(Object.keys(res.body.adversarial_results)).toHaveLength(2);
  });

  test("rejects missing y", async () => {
    const res = await request(app)
      .post("/api/adversarial/evaluate")
      .send({ x: [0.5] });
    expect(res.status).toBe(400);
  });
});

describe("POST /api/adversarial/attack", () => {
  test("valid attack", async () => {
    const res = await request(app)
      .post("/api/adversarial/attack")
      .send({ x: [0.5, 0.3], y: 0, epsilon: 0.1 });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("x_adv");
    expect(res.body).toHaveProperty("perturbation_norm");
    expect(res.body).toHaveProperty("success");
    expect(res.body.x_adv).toHaveLength(2);
  });

  test("rejects negative epsilon", async () => {
    const res = await request(app)
      .post("/api/adversarial/attack")
      .send({ x: [0.5], y: 0, epsilon: -0.1 });
    expect(res.status).toBe(400);
  });
});

describe("POST /api/active/init + step", () => {
  let sessionId: string;

  test("init creates session", async () => {
    const pool = Array.from({ length: 20 }, (_, i) => [
      Math.sin(i),
      Math.cos(i),
    ]);
    const res = await request(app)
      .post("/api/active/init")
      .send({ pool, w_uncertainty: 0.7, w_diversity: 0.3 });
    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty("session_id");
    expect(res.body.pool_size).toBe(20);
    sessionId = res.body.session_id;
  });

  test("step selects batch", async () => {
    const res = await request(app)
      .post("/api/active/step")
      .send({ session_id: sessionId, k: 3 });
    expect(res.status).toBe(200);
    expect(res.body.selected_indices).toHaveLength(3);
    expect(res.body.scores).toHaveLength(3);
    expect(res.body.done).toBe(false);
  });

  test("step with unknown session", async () => {
    const res = await request(app)
      .post("/api/active/step")
      .send({ session_id: "nonexistent", k: 3 });
    expect(res.status).toBe(404);
  });

  test("rejects invalid k", async () => {
    const res = await request(app)
      .post("/api/active/step")
      .send({ session_id: sessionId, k: -1 });
    expect(res.status).toBe(400);
  });
});

describe("POST /api/drift/scenario", () => {
  test("stable distribution", async () => {
    const data = Array.from({ length: 50 }, () => [
      Math.random(),
      Math.random(),
    ]);
    const res = await request(app)
      .post("/api/drift/scenario")
      .send({ reference: data, current: data });
    expect(res.status).toBe(200);
    expect(res.body.phase).toBe("STABLE");
    expect(res.body.kl_score).toBeCloseTo(0, 1);
  });

  test("drifted distribution", async () => {
    // Use spread-out distributions to ensure binning captures the shift
    const ref = Array.from({ length: 50 }, (_, i) => [i * 0.1, i * 0.1]);
    const cur = Array.from({ length: 50 }, (_, i) => [i * 0.1 + 5, i * 0.1 + 5]);
    const res = await request(app)
      .post("/api/drift/scenario")
      .send({ reference: ref, current: cur });
    expect(res.status).toBe(200);
    expect(["DRIFT", "CRITICAL"]).toContain(res.body.phase);
    expect(res.body.kl_score).toBeGreaterThan(0);
  });

  test("with custom window size", async () => {
    const data = Array.from({ length: 30 }, () => [Math.random()]);
    const res = await request(app)
      .post("/api/drift/scenario")
      .send({ reference: data, current: data, window_size: 10 });
    expect(res.status).toBe(200);
    expect(res.body.window_size).toBe(10);
  });

  test("rejects missing data", async () => {
    const res = await request(app).post("/api/drift/scenario").send({});
    expect(res.status).toBe(400);
  });
});

// ═══════════════════════════════════════════════
//  WEBSOCKET TESTS
// ═══════════════════════════════════════════════

describe("WebSocket /ws/optimize", () => {
  test("full optimization run", (done) => {
    const ws = new WebSocket(`ws://localhost:${PORT_TEST}`);
    const steps: unknown[] = [];

    ws.on("open", () => {
      ws.send(
        JSON.stringify({
          type: "START",
          function: "ackley",
          optimizer: "standard",
          n_iterations: 5,
        })
      );
    });

    ws.on("message", (data: Buffer) => {
      const msg = JSON.parse(data.toString());
      if (msg.type === "STEP") {
        steps.push(msg);
        expect(msg).toHaveProperty("step");
        expect(msg).toHaveProperty("candidate");
        expect(msg).toHaveProperty("observed");
        expect(msg).toHaveProperty("best_so_far");
        expect(msg).toHaveProperty("acquisition_value");
      }
      if (msg.type === "COMPLETE") {
        expect(msg).toHaveProperty("best_x");
        expect(msg).toHaveProperty("best_y");
        expect(msg).toHaveProperty("total_steps");
        expect(msg.total_steps).toBe(5);
        expect(msg).toHaveProperty("convergence_history");
        expect(msg.convergence_history).toHaveLength(5);
        expect(steps).toHaveLength(5);
        ws.close();
        done();
      }
    });
  }, 15000);

  test("PRIME optimizer includes prime_state", (done) => {
    const ws = new WebSocket(`ws://localhost:${PORT_TEST}`);

    ws.on("open", () => {
      ws.send(
        JSON.stringify({
          type: "START",
          function: "rastrigin",
          optimizer: "prime",
          n_iterations: 3,
        })
      );
    });

    ws.on("message", (data: Buffer) => {
      const msg = JSON.parse(data.toString());
      if (msg.type === "STEP") {
        expect(msg).toHaveProperty("prime_state");
        expect(msg.prime_state).toHaveProperty("phase");
        expect(msg.prime_state).toHaveProperty("variance");
        expect(["CONVERGING", "EXPLORING", "BIFURCATING"]).toContain(
          msg.prime_state.phase
        );
      }
      if (msg.type === "COMPLETE") {
        ws.close();
        done();
      }
    });
  }, 15000);

  test("invalid message returns ERROR", (done) => {
    const ws = new WebSocket(`ws://localhost:${PORT_TEST}`);

    ws.on("open", () => {
      ws.send(JSON.stringify({ type: "INVALID" }));
    });

    ws.on("message", (data: Buffer) => {
      const msg = JSON.parse(data.toString());
      expect(msg.type).toBe("ERROR");
      expect(msg.message).toBeDefined();
      ws.close();
      done();
    });
  }, 5000);

  test("malformed JSON returns ERROR", (done) => {
    const ws = new WebSocket(`ws://localhost:${PORT_TEST}`);

    ws.on("open", () => {
      ws.send("this is not json");
    });

    ws.on("message", (data: Buffer) => {
      const msg = JSON.parse(data.toString());
      expect(msg.type).toBe("ERROR");
      ws.close();
      done();
    });
  }, 5000);

  test("custom bounds", (done) => {
    const ws = new WebSocket(`ws://localhost:${PORT_TEST}`);

    ws.on("open", () => {
      ws.send(
        JSON.stringify({
          type: "START",
          function: "himmelblau",
          n_iterations: 2,
          bounds: [
            [-6, 6],
            [-6, 6],
          ],
        })
      );
    });

    ws.on("message", (data: Buffer) => {
      const msg = JSON.parse(data.toString());
      if (msg.type === "COMPLETE") {
        expect(msg.total_steps).toBe(2);
        ws.close();
        done();
      }
    });
  }, 10000);
});

// ═══════════════════════════════════════════════
//  VALIDATION EDGE CASES
// ═══════════════════════════════════════════════

describe("Validation edge cases", () => {
  test("empty body defaults handled", async () => {
    // These should fail validation since required fields are missing
    const endpoints = [
      "/api/classify",
      "/api/regress",
      "/api/openset",
      "/api/adversarial/evaluate",
      "/api/adversarial/attack",
      "/api/drift/scenario",
    ];
    for (const ep of endpoints) {
      const res = await request(app).post(ep).send({});
      expect(res.status).toBe(400);
    }
  });

  test("extra fields are ignored by Zod strip", async () => {
    const res = await request(app)
      .post("/api/classify")
      .send({ x: [0.1], extra_field: "ignored" });
    expect(res.status).toBe(200);
  });
});
