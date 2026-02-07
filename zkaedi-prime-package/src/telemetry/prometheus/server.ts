/**
 * 🔱 ZKAEDI PRIME — Prometheus HTTP Server
 * 
 * HTTP server for Prometheus metrics scraping
 */

import express, { Express, Request, Response } from 'express';
import { Server } from 'http';
import { PrometheusExporter, getPrometheusExporter } from './PrometheusExporter';
import { PrometheusConfig } from '../../config';

/**
 * Prometheus HTTP Server for metrics scraping
 */
export class PrometheusServer {
  private app: Express;
  private server?: Server;
  private exporter: PrometheusExporter;
  private config: Required<PrometheusConfig>;

  constructor(exporter?: PrometheusExporter, config: PrometheusConfig = {}) {
    this.exporter = exporter ?? getPrometheusExporter();
    this.config = {
      enabled: config.enabled ?? true,
      port: config.port ?? 9090,
      path: config.path ?? '/metrics',
    };

    this.app = express();
    this.setupRoutes();
  }

  /**
   * Setup routes
   */
  private setupRoutes(): void {
    // Metrics endpoint
    this.app.get(this.config.path, async (_req: Request, res: Response) => {
      try {
        res.set('Content-Type', this.exporter.getRegister().contentType);
        const metrics = await this.exporter.getMetrics();
        res.end(metrics);
      } catch (error) {
        res.status(500).send(error instanceof Error ? error.message : 'Unknown error');
      }
    });

    // Health check
    this.app.get('/health', (_req: Request, res: Response) => {
      res.json({ status: 'healthy', timestamp: new Date().toISOString() });
    });

    // Readiness check
    this.app.get('/ready', (_req: Request, res: Response) => {
      res.json({ status: 'ready', timestamp: new Date().toISOString() });
    });
  }

  /**
   * Start the server
   */
  async start(): Promise<void> {
    if (!this.config.enabled) {
      console.log('Prometheus server is disabled');
      return;
    }

    return new Promise((resolve) => {
      this.server = this.app.listen(this.config.port, () => {
        console.log(`Prometheus metrics server listening on port ${this.config.port}`);
        console.log(`Metrics available at http://localhost:${this.config.port}${this.config.path}`);
        resolve();
      });
    });
  }

  /**
   * Stop the server
   */
  async stop(): Promise<void> {
    if (!this.server) {
      return;
    }

    return new Promise((resolve, reject) => {
      this.server?.close((err) => {
        if (err) {
          reject(err);
        } else {
          console.log('Prometheus metrics server stopped');
          resolve();
        }
      });
    });
  }

  /**
   * Get the Express app
   */
  getApp(): Express {
    return this.app;
  }
}
