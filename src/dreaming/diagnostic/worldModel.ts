import type { Rng } from '../rng';
import { DiagnosticWorkupEnv } from './env';
import { Test, TestResult, type DiagnosticConfig } from './types';

export interface DiagnosticTransitionSample {
  s: number;
  a: number;
  r: number;
  sNext: number;
  done: boolean;
}

/** Empirical (s,a) -> (s', r, done) model learned from real rollouts. */
export class DiagnosticWorldModel {
  private outcomes = new Map<string, Map<string, { count: number; rSum: number }>>();
  private observedPairs: string[] = [];

  private pairKey(s: number, a: number): string {
    return `${s},${a}`;
  }

  private outcomeKey(sNext: number, done: boolean): string {
    return `${sNext},${done ? 1 : 0}`;
  }

  update(s: number, a: number, r: number, sNext: number, done: boolean): void {
    const key = this.pairKey(s, a);
    if (!this.outcomes.has(key)) {
      this.outcomes.set(key, new Map());
      this.observedPairs.push(key);
    }
    const bucket = this.outcomes.get(key)!;
    const oKey = this.outcomeKey(sNext, done);
    const prev = bucket.get(oKey) ?? { count: 0, rSum: 0 };
    bucket.set(oKey, { count: prev.count + 1, rSum: prev.rSum + r });
  }

  sample(rng: Rng): DiagnosticTransitionSample | null {
    if (this.observedPairs.length === 0) return null;
    const key = this.observedPairs[Math.floor(rng() * this.observedPairs.length)]!;
    const [sStr, aStr] = key.split(',');
    const s = Number(sStr);
    const a = Number(aStr);
    const bucket = this.outcomes.get(key)!;
    const entries = [...bucket.entries()];
    const total = entries.reduce((sum, [, v]) => sum + v.count, 0);
    let roll = rng() * total;
    let chosen = entries[0]!;
    for (const entry of entries) {
      roll -= entry[1].count;
      if (roll <= 0) {
        chosen = entry;
        break;
      }
    }
    const [oKey, stats] = chosen;
    const [sNextStr, doneStr] = oKey.split(',');
    return {
      s,
      a,
      r: stats.rSum / stats.count,
      sNext: Number(sNextStr),
      done: doneStr === '1',
    };
  }

  get observedCount(): number {
    return this.observedPairs.length;
  }
}

/**
 * Biased learned model: underestimates sensitivity of one test when sampling
 * imagined positive findings. Creates clinically interpretable model error.
 */
export class BiasedDiagnosticWorldModel extends DiagnosticWorldModel {
  constructor(
    private env: DiagnosticWorkupEnv,
    private config: DiagnosticConfig
  ) {
    super();
  }

  override sample(rng: Rng): DiagnosticTransitionSample | null {
    const base = super.sample(rng);
    if (!base || this.config.modelBiasLevel <= 0) return base;

    const testAction = base.a;
    if (testAction < 0 || testAction > 3) return base;

    const test = testAction as Test;
    if (test !== this.config.biasedTest) return base;

    const state = this.env.decode(base.s);
    const prev = state.results[test];
    if (prev !== TestResult.Unobserved) return base;

    // With probability proportional to bias, flip positive findings to negative
    // in the learned model (underestimating sensitivity).
    if (base.r < 0 && rng() < this.config.modelBiasLevel * 0.6) {
      const biasedState = {
        ...state,
        results: { ...state.results, [test]: TestResult.Negative },
      };
      base.sNext = this.env.encode(biasedState);
      base.r = -this.config.testCosts[test] - this.config.delayPenalty;
      base.done = false;
    }
    return base;
  }
}

export function makeDiagnosticModel(
  env: DiagnosticWorkupEnv,
  config: DiagnosticConfig
): DiagnosticWorldModel {
  if (config.modelBiasLevel > 0) {
    return new BiasedDiagnosticWorldModel(env, config);
  }
  return new DiagnosticWorldModel();
}
