/**
 * 🔱 ZKAEDI PRIME — Integration Configuration
 * 
 * Configuration interfaces for all ecosystem integrations
 */

export interface TensorFlowJSConfig {
  backend?: 'webgl' | 'cpu' | 'wasm';
  enableProfiler?: boolean;
  memoryLimit?: number;
}

export interface MLflowConfig {
  trackingUri?: string;
  experimentName?: string;
  enableAutoLogging?: boolean;
  authentication?: {
    type: 'basic' | 'token';
    credentials?: {
      username?: string;
      password?: string;
      token?: string;
    };
  };
}

export interface OpenTelemetryConfig {
  serviceName?: string;
  endpoint?: string;
  enableTracing?: boolean;
  enableMetrics?: boolean;
  samplingRate?: number;
}

export interface PrometheusConfig {
  enabled?: boolean;
  port?: number;
  path?: string;
}

export interface WandbConfig {
  project?: string;
  entity?: string;
  apiKey?: string;
  mode?: 'online' | 'offline' | 'disabled';
}

export interface ServerConfig {
  enabled?: boolean;
  port?: number;
  host?: string;
  authentication?: {
    enabled?: boolean;
    type?: 'apikey' | 'jwt' | 'oauth';
  };
}

/**
 * Main integration configuration interface
 */
export interface IntegrationConfig {
  tensorflowjs?: TensorFlowJSConfig;
  mlflow?: MLflowConfig;
  opentelemetry?: OpenTelemetryConfig;
  prometheus?: PrometheusConfig;
  wandb?: WandbConfig;
  server?: ServerConfig;
}

/**
 * Default configuration values
 */
export const DEFAULT_CONFIG: Required<IntegrationConfig> = {
  tensorflowjs: {
    backend: 'cpu',
    enableProfiler: false,
    memoryLimit: undefined,
  },
  mlflow: {
    trackingUri: 'http://localhost:5000',
    experimentName: 'zkaedi-prime-experiment',
    enableAutoLogging: true,
    authentication: undefined,
  },
  opentelemetry: {
    serviceName: 'zkaedi-prime',
    endpoint: 'http://localhost:4318',
    enableTracing: true,
    enableMetrics: true,
    samplingRate: 1.0,
  },
  prometheus: {
    enabled: false,
    port: 9090,
    path: '/metrics',
  },
  wandb: {
    project: 'zkaedi-prime',
    entity: undefined,
    apiKey: undefined,
    mode: 'disabled',
  },
  server: {
    enabled: false,
    port: 3000,
    host: '0.0.0.0',
    authentication: {
      enabled: false,
      type: 'apikey',
    },
  },
};

/**
 * Merge user config with defaults
 */
export function mergeConfig(userConfig?: Partial<IntegrationConfig>): IntegrationConfig {
  if (!userConfig) {
    return DEFAULT_CONFIG;
  }

  return {
    tensorflowjs: { ...DEFAULT_CONFIG.tensorflowjs, ...userConfig.tensorflowjs },
    mlflow: { ...DEFAULT_CONFIG.mlflow, ...userConfig.mlflow },
    opentelemetry: { ...DEFAULT_CONFIG.opentelemetry, ...userConfig.opentelemetry },
    prometheus: { ...DEFAULT_CONFIG.prometheus, ...userConfig.prometheus },
    wandb: { ...DEFAULT_CONFIG.wandb, ...userConfig.wandb },
    server: { ...DEFAULT_CONFIG.server, ...userConfig.server },
  };
}
