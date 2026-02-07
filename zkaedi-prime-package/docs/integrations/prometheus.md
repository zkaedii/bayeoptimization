# 📊 Prometheus Metrics Integration

## Overview

ZKAEDI PRIME provides comprehensive Prometheus metrics export for monitoring optimization runs, tracking performance, and alerting.

## Quick Start

```typescript
import { PrometheusExporter, PrometheusServer } from '@zkaedi/zkaedi-prime/telemetry';

// Create exporter
const exporter = new PrometheusExporter({ enabled: true });

// Start HTTP server
const server = new PrometheusServer(exporter, {
  enabled: true,
  port: 9090,
  path: '/metrics',
});

await server.start();
```

## Standard Metrics

ZKAEDI PRIME exports these metrics automatically:

- `zkaedi_optimization_duration_seconds` - Optimization duration
- `zkaedi_optimization_iterations_total` - Total iterations
- `zkaedi_objective_evaluations_total` - Objective function calls
- `zkaedi_gp_training_time_seconds` - GP training time
- `zkaedi_acquisition_function_time_seconds` - Acquisition function time
- `zkaedi_memory_usage_bytes` - Memory usage
- `zkaedi_errors_total` - Error count

## Usage

### Basic Metrics

```typescript
const exporter = new PrometheusExporter();

// Record optimization
exporter.optimizationIterations.inc({ status: 'success' });
exporter.optimizationDuration.observe({ status: 'success' }, 125.5);

// Track memory
exporter.memoryUsage.set(process.memoryUsage().heapUsed);

// Count evaluations
exporter.objectiveEvaluations.inc();
```

### Custom Metrics

```typescript
const customCounter = exporter.createCustomMetric(
  'my_custom_metric_total',
  'counter',
  'Description of my metric',
  ['label1', 'label2']
);

customCounter.inc({ label1: 'value1', label2: 'value2' });
```

### HTTP Server

```typescript
const server = new PrometheusServer(exporter, {
  enabled: true,
  port: 9090,
  path: '/metrics',
});

await server.start();
// Metrics at: http://localhost:9090/metrics
// Health at: http://localhost:9090/health

await server.stop();
```

## Prometheus Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'zkaedi-prime'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 15s
```

## Grafana Dashboard

Create visualizations:

1. Add Prometheus as data source
2. Import metrics
3. Create panels for key metrics
4. Set up alerts

## Best Practices

- Use labels for filtering
- Set appropriate scrape intervals
- Monitor memory usage
- Set up alerts for errors

## Examples

See [examples/integrations/prometheus-metrics-example.ts](../../examples/integrations/prometheus-metrics-example.ts)
