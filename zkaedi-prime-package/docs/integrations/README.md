# 🔗 ZKAEDI PRIME Ecosystem Integrations

## Overview

ZKAEDI PRIME Phase 1 provides comprehensive ecosystem integrations for production-grade ML/AI workflows. These integrations enable:

- **Experiment Tracking** with MLflow
- **Observability** with OpenTelemetry  
- **Metrics** with Prometheus
- **Model Serving** with REST API
- **Interoperability** with TensorFlow.js, ONNX, and Python

## Available Integrations

### Core Integrations

- [MLflow](./mlflow.md) - Experiment tracking and model registry
- [OpenTelemetry](./opentelemetry.md) - Distributed tracing and observability
- [Prometheus](./prometheus.md) - Metrics collection and monitoring

### Quick Start

```typescript
import { BayesianOptimizer } from '@zkaedi/zkaedi-prime';
import { MLflowTracker } from '@zkaedi/zkaedi-prime/integrations';
import { PrometheusExporter } from '@zkaedi/zkaedi-prime/telemetry';

// MLflow tracking
const mlflow = new MLflowTracker();
await mlflow.startRun('my-run');

// Prometheus metrics
const prometheus = new PrometheusExporter();

// Run optimization
const optimizer = new BayesianOptimizer({ x: [-5, 5] });
const result = await optimizer.optimize(objective);

// Log results
await mlflow.logMetric('best_value', result.bestY);
await mlflow.endRun('FINISHED');
```

## Examples

See [examples/integrations/](../../examples/integrations/) for complete examples.

## Documentation

- Configuration: See [src/config/IntegrationConfig.ts](../../src/config/IntegrationConfig.ts)
- Individual guides in this directory
