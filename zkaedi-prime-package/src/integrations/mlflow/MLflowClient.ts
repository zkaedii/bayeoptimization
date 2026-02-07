/**
 * 🔱 ZKAEDI PRIME — MLflow Client
 * 
 * HTTP client for MLflow tracking server REST API
 */

import axios, { AxiosInstance, AxiosRequestConfig } from 'axios';
import { MLflowConfig } from '../../config';

export interface MLflowExperiment {
  experiment_id: string;
  name: string;
  artifact_location: string;
  lifecycle_stage: string;
}

export interface MLflowRun {
  run_id: string;
  experiment_id: string;
  status: string;
  start_time: number;
  end_time?: number;
  artifact_uri: string;
}

/**
 * MLflow REST API Client
 */
export class MLflowClient {
  private client: AxiosInstance;
  private config: Required<Omit<MLflowConfig, 'authentication'>>;

  constructor(config: MLflowConfig = {}) {
    this.config = {
      trackingUri: config.trackingUri ?? 'http://localhost:5000',
      experimentName: config.experimentName ?? 'zkaedi-prime',
      enableAutoLogging: config.enableAutoLogging ?? true,
    };

    const axiosConfig: AxiosRequestConfig = {
      baseURL: this.config.trackingUri,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    // Add authentication if provided
    if (config.authentication) {
      if (config.authentication.type === 'basic') {
        const { username, password } = config.authentication.credentials ?? {};
        if (username && password) {
          axiosConfig.auth = { username, password };
        }
      } else if (config.authentication.type === 'token') {
        const { token } = config.authentication.credentials ?? {};
        if (token) {
          axiosConfig.headers = {
            ...axiosConfig.headers,
            Authorization: `Bearer ${token}`,
          };
        }
      }
    }

    this.client = axios.create(axiosConfig);
  }

  /**
   * Create a new experiment
   */
  async createExperiment(name: string, artifactLocation?: string): Promise<string> {
    try {
      const response = await this.client.post('/api/2.0/mlflow/experiments/create', {
        name,
        artifact_location: artifactLocation,
      });
      return response.data.experiment_id;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 400) {
        // Experiment already exists, get it
        const experiment = await this.getExperimentByName(name);
        return experiment.experiment_id;
      }
      throw error;
    }
  }

  /**
   * Get experiment by name
   */
  async getExperimentByName(name: string): Promise<MLflowExperiment> {
    const response = await this.client.get('/api/2.0/mlflow/experiments/get-by-name', {
      params: { experiment_name: name },
    });
    return response.data.experiment;
  }

  /**
   * Create a new run
   */
  async createRun(
    experimentId: string,
    runName?: string,
    tags?: Record<string, string>
  ): Promise<MLflowRun> {
    const response = await this.client.post('/api/2.0/mlflow/runs/create', {
      experiment_id: experimentId,
      run_name: runName,
      tags: tags ? Object.entries(tags).map(([key, value]) => ({ key, value })) : [],
    });
    return response.data.run.info;
  }

  /**
   * Update run
   */
  async updateRun(
    runId: string,
    status: 'RUNNING' | 'SCHEDULED' | 'FINISHED' | 'FAILED' | 'KILLED',
    endTime?: number
  ): Promise<void> {
    await this.client.post('/api/2.0/mlflow/runs/update', {
      run_id: runId,
      status,
      end_time: endTime ?? Date.now(),
    });
  }

  /**
   * Log metric
   */
  async logMetric(
    runId: string,
    key: string,
    value: number,
    timestamp?: number,
    step?: number
  ): Promise<void> {
    await this.client.post('/api/2.0/mlflow/runs/log-metric', {
      run_id: runId,
      key,
      value,
      timestamp: timestamp ?? Date.now(),
      step: step ?? 0,
    });
  }

  /**
   * Log parameter
   */
  async logParam(runId: string, key: string, value: string): Promise<void> {
    await this.client.post('/api/2.0/mlflow/runs/log-parameter', {
      run_id: runId,
      key,
      value: String(value),
    });
  }

  /**
   * Set tag
   */
  async setTag(runId: string, key: string, value: string): Promise<void> {
    await this.client.post('/api/2.0/mlflow/runs/set-tag', {
      run_id: runId,
      key,
      value,
    });
  }

  /**
   * Log batch
   */
  async logBatch(
    runId: string,
    metrics?: Array<{ key: string; value: number; timestamp?: number; step?: number }>,
    params?: Array<{ key: string; value: string }>,
    tags?: Array<{ key: string; value: string }>
  ): Promise<void> {
    const payload: any = { run_id: runId };

    if (metrics) {
      payload.metrics = metrics.map((m) => ({
        key: m.key,
        value: m.value,
        timestamp: m.timestamp ?? Date.now(),
        step: m.step ?? 0,
      }));
    }

    if (params) {
      payload.params = params.map((p) => ({ key: p.key, value: String(p.value) }));
    }

    if (tags) {
      payload.tags = tags;
    }

    await this.client.post('/api/2.0/mlflow/runs/log-batch', payload);
  }
}
