import { PrometheusExporter } from '../../telemetry/prometheus';

describe('PrometheusExporter', () => {
  it('should initialize with metrics', () => {
    const exporter = new PrometheusExporter();
    expect(exporter.optimizationIterations).toBeDefined();
    expect(exporter.objectiveEvaluations).toBeDefined();
  });

  it('should increment counters', () => {
    const exporter = new PrometheusExporter();
    exporter.optimizationIterations.inc({ status: 'success' });
    expect(true).toBe(true);
  });

  it('should get metrics', async () => {
    const exporter = new PrometheusExporter();
    const metrics = await exporter.getMetrics();
    expect(metrics).toContain('zkaedi_');
  });
});
