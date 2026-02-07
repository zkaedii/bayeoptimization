# 🔍 OpenTelemetry Integration

## Overview

ZKAEDI PRIME integrates with OpenTelemetry for distributed tracing, metrics collection, and comprehensive observability.

## Quick Start

```typescript
import { TelemetryProvider } from '@zkaedi/zkaedi-prime/telemetry';

const telemetry = new TelemetryProvider({
  serviceName: 'zkaedi-prime',
  endpoint: 'http://localhost:4318',
  enableTracing: true,
});

telemetry.initialize();

// Use with spans
await telemetry.withSpan('optimization', async (span) => {
  // Your code here
  return result;
});

await telemetry.shutdown();
```

## Features

- **Distributed Tracing**: Track operations across services
- **Context Propagation**: Maintain context through async calls
- **Span Attributes**: Add metadata to traces
- **Status Tracking**: Success/error status on spans

## Usage

### Basic Tracing

```typescript
const span = telemetry.startSpan('my-operation', {
  attribute1: 'value1',
  attribute2: 42,
});

try {
  // Your code
  telemetry.endSpan(span, { code: SpanStatusCode.OK });
} catch (error) {
  telemetry.endSpan(span, { 
    code: SpanStatusCode.ERROR,
    message: error.message 
  });
}
```

### Helper Method

```typescript
await telemetry.withSpan('operation', async (span) => {
  // Automatically handles success/error
  return result;
}, { metadata: 'value' });
```

### Span Events

```typescript
telemetry.addSpanEvent(span, 'checkpoint', {
  progress: 0.5,
  timestamp: Date.now(),
});
```

## OpenTelemetry Backends

Compatible with:
- Jaeger
- Zipkin
- Grafana Tempo
- AWS X-Ray
- Google Cloud Trace

## Configuration

```typescript
const telemetry = new TelemetryProvider({
  serviceName: 'my-service',
  endpoint: 'http://otel-collector:4318',
  enableTracing: true,
  enableMetrics: true,
  samplingRate: 1.0, // 100% sampling
});
```

## Best Practices

- Add meaningful span names
- Use attributes for filtering
- Keep sampling rate appropriate
- Propagate context correctly

## Examples

See [examples/integrations/opentelemetry-tracing-example.ts](../../examples/integrations/opentelemetry-tracing-example.ts)
