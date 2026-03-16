/**
 * ZKAEDI PRIME - Test Functions for Bayesian Optimization
 *
 * Standard benchmark optimization functions. All accept number arrays
 * and return scalar values. Designed for minimization — the global
 * minimum of each function is documented below.
 */

/**
 * Ackley function — multimodal with many local minima.
 *
 * Global minimum: f(0, 0, ..., 0) ≈ 0
 *
 * @param x - Input vector of any dimensionality
 * @returns Scalar function value
 *
 * @example
 * ```ts
 * ackley([0, 0]); // ≈ 4.44e-16 (effectively 0)
 * ```
 */
export function ackley(x: number[]): number {
  const n = x.length;
  const a = 20;
  const b = 0.2;
  const c = 2 * Math.PI;

  let sumSq = 0;
  let sumCos = 0;
  for (let i = 0; i < n; i++) {
    sumSq += x[i] * x[i];
    sumCos += Math.cos(c * x[i]);
  }

  const term1 = -a * Math.exp(-b * Math.sqrt(sumSq / n));
  const term2 = -Math.exp(sumCos / n);
  return term1 + term2 + a + Math.E;
}

/**
 * Rastrigin function — highly multimodal with regular oscillations.
 *
 * Global minimum: f(0, 0, ..., 0) = 0
 *
 * @param x - Input vector of any dimensionality
 * @returns Scalar function value
 *
 * @example
 * ```ts
 * rastrigin([0, 0]); // 0
 * ```
 */
export function rastrigin(x: number[]): number {
  const A = 10;
  const n = x.length;
  let sum = A * n;
  for (let i = 0; i < n; i++) {
    sum += x[i] * x[i] - A * Math.cos(2 * Math.PI * x[i]);
  }
  return sum;
}

/**
 * Rosenbrock function — banana-shaped valley, hard to converge.
 *
 * Global minimum: f(1, 1, ..., 1) = 0
 *
 * @param x - Input vector (dimensionality ≥ 2)
 * @returns Scalar function value
 *
 * @example
 * ```ts
 * rosenbrock([1, 1]); // 0
 * ```
 */
export function rosenbrock(x: number[]): number {
  let sum = 0;
  for (let i = 0; i < x.length - 1; i++) {
    sum += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2;
  }
  return sum;
}

/**
 * Schwefel function — deceptive, global minimum far from local minima.
 *
 * Global minimum: f(420.9687, ..., 420.9687) ≈ 0
 *
 * @param x - Input vector of any dimensionality
 * @returns Scalar function value
 *
 * @example
 * ```ts
 * schwefel([420.9687, 420.9687]); // ≈ 0
 * ```
 */
export function schwefel(x: number[]): number {
  const n = x.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += x[i] * Math.sin(Math.sqrt(Math.abs(x[i])));
  }
  return 418.9829 * n - sum;
}

/**
 * Himmelblau function — four symmetric global minima (2D only).
 *
 * Global minima: f(3,2) = f(-2.805,3.131) = f(-3.779,-3.283) = f(3.584,-1.848) = 0
 *
 * @param x - Input vector (must be exactly 2D)
 * @returns Scalar function value
 *
 * @example
 * ```ts
 * himmelblau([3, 2]); // 0
 * ```
 */
export function himmelblau(x: number[]): number {
  const [a, b] = x;
  return (a * a + b - 11) ** 2 + (a + b * b - 7) ** 2;
}

/** Map of function names to implementations. */
export const TEST_FUNCTIONS: Record<string, (x: number[]) => number> = {
  ackley,
  rastrigin,
  rosenbrock,
  schwefel,
  himmelblau,
};
