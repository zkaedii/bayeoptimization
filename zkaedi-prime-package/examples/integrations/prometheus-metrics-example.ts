/**
 * 🔱 ZKAEDI PRIME — Prometheus Metrics Example
 */

import { BayesianOptimizer } from '../../src/optimization';
import { PrometheusExporter, PrometheusServer } from '../../src/telemetry/prometheus';

async function main() {
  console.log('🔱 ZKAEDI PRIME — Prometheus Metrics Example\n');

  const exporter = new PrometheusExporter({ enabled: true });
  const server = new PrometheusServer(exporter, { enabled: true, port: 9090 });

  await server.start();
  console.log('✅ Prometheus server started at http://localhost:9090/metrics\n');

  const optimizer = new BayesianOptimizer(
    { x: [-5, 5], y: [-5, 5] },
    { nIter: 30, nWarmup: 10 }
  );

  const objective = async (params: number[]): Promise<number> => {
    const [x, y] = params;
    exporter.objectiveEvaluations.inc();
    exporter.memoryUsage.set(process.memoryUsage().heapUsed);
    return -(x ** 2 + y ** 2);
  };

  console.log('🚀 Starting optimization with metrics...\n');
  const result = await optimizer.optimize(objective);

  console.log(`✅ Best: [${result.bestX.join(', ')}] = ${result.bestY}`);
  console.log('\n📊 Metrics: http://localhost:9090/metrics');

  await new Promise(r => setTimeout(r, 30000));
  await server.stop();
}

main().catch(console.error);
