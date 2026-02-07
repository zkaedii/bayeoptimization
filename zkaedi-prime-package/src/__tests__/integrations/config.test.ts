import { mergeConfig, DEFAULT_CONFIG } from '../../config';

describe('IntegrationConfig', () => {
  it('should use default config', () => {
    const config = mergeConfig();
    expect(config).toEqual(DEFAULT_CONFIG);
  });

  it('should merge user config', () => {
    const config = mergeConfig({
      mlflow: { trackingUri: 'http://custom:5000' },
    });
    expect(config.mlflow?.trackingUri).toBe('http://custom:5000');
  });
});
