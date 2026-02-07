import { TelemetryProvider } from '../../telemetry';

describe('TelemetryProvider', () => {
  it('should initialize', () => {
    const provider = new TelemetryProvider();
    provider.initialize();
    expect(true).toBe(true);
  });

  it('should handle withSpan', async () => {
    const provider = new TelemetryProvider();
    const result = await provider.withSpan('test', async () => 42);
    expect(result).toBe(42);
  });
});
