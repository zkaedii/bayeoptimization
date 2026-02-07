/**
 * 🔱 ZKAEDI PRIME — OpenTelemetry Provider
 * 
 * Comprehensive observability with OpenTelemetry
 */

import { 
  trace, 
  Span, 
  SpanStatusCode, 
  Tracer,
  Context,
  context as otelContext
} from '@opentelemetry/api';
import { OpenTelemetryConfig } from '../config';

export interface SpanStatus {
  code: SpanStatusCode;
  message?: string;
}

/**
 * Telemetry Provider for OpenTelemetry instrumentation
 */
export class TelemetryProvider {
  private tracer?: Tracer;
  private config: OpenTelemetryConfig;
  private initialized = false;

  constructor(config: OpenTelemetryConfig = {}) {
    this.config = {
      serviceName: config.serviceName ?? 'zkaedi-prime',
      endpoint: config.endpoint ?? 'http://localhost:4318',
      enableTracing: config.enableTracing ?? true,
      enableMetrics: config.enableMetrics ?? true,
      samplingRate: config.samplingRate ?? 1.0,
    };
  }

  /**
   * Initialize the telemetry provider
   */
  initialize(serviceName?: string): void {
    if (this.initialized) {
      return;
    }

    if (serviceName) {
      this.config.serviceName = serviceName;
    }

    if (this.config.enableTracing) {
      this.tracer = trace.getTracer(this.config.serviceName ?? 'zkaedi-prime');
    }

    this.initialized = true;
  }

  /**
   * Shutdown the telemetry provider
   */
  async shutdown(): Promise<void> {
    this.initialized = false;
  }

  /**
   * Start a new span
   */
  startSpan(name: string, attributes?: Record<string, any>): Span | null {
    if (!this.tracer || !this.config.enableTracing) {
      return null;
    }

    const span = this.tracer.startSpan(name, {
      attributes: attributes ?? {},
    });

    return span;
  }

  /**
   * End a span
   */
  endSpan(span: Span | null, status?: SpanStatus): void {
    if (!span) {
      return;
    }

    if (status) {
      span.setStatus(status);
    }

    span.end();
  }

  /**
   * Add an event to a span
   */
  addSpanEvent(span: Span | null, name: string, attributes?: Record<string, any>): void {
    if (!span) {
      return;
    }

    span.addEvent(name, attributes);
  }

  /**
   * Get the current context
   */
  getCurrentContext(): Context {
    return otelContext.active();
  }

  /**
   * Set the current context
   */
  setContext(context: Context): void {
    otelContext.with(context, () => {
      // Context is now active
    });
  }

  /**
   * Execute function with span context
   */
  async withSpan<T>(
    name: string,
    fn: (span: Span | null) => Promise<T>,
    attributes?: Record<string, any>
  ): Promise<T> {
    const span = this.startSpan(name, attributes);

    try {
      const result = await fn(span);
      this.endSpan(span, { code: SpanStatusCode.OK });
      return result;
    } catch (error) {
      this.endSpan(span, {
        code: SpanStatusCode.ERROR,
        message: error instanceof Error ? error.message : String(error),
      });
      throw error;
    }
  }

  /**
   * Check if telemetry is enabled
   */
  isEnabled(): boolean {
    return this.initialized && (this.config.enableTracing ?? false);
  }
}

/**
 * Global telemetry provider instance
 */
let globalProvider: TelemetryProvider | null = null;

/**
 * Get or create the global telemetry provider
 */
export function getTelemetryProvider(config?: OpenTelemetryConfig): TelemetryProvider {
  if (!globalProvider) {
    globalProvider = new TelemetryProvider(config);
  }
  return globalProvider;
}

/**
 * Reset the global telemetry provider
 */
export function resetTelemetryProvider(): void {
  globalProvider = null;
}
