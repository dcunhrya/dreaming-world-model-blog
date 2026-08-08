import { describe, expect, it } from 'vitest';
import { DEFAULT_DIAGNOSTIC_CONFIG } from './config';
import { DiagnosticWorkupEnv } from './env';
import { DiagnosticWorldModel } from './worldModel';
import {
  decodeState,
  encodeState,
  initialInformationState,
  NUM_INFORMATION_STATES,
} from './state';
import {
  Disease,
  RiskProfile,
  Test,
  TestResult,
  WorkupAction,
} from './types';
import { mulberry32 } from '../rng';
import { runDiagnosticSingle } from './training';
import { diagnosticResultsChecksum, runDiagnosticOfflineExperiments } from './runExperiment';
import { OFFLINE_DIAGNOSTIC_PROFILE } from './config';

describe('diagnostic state encoding', () => {
  it('round-trips all state ids', () => {
    for (let id = 0; id < NUM_INFORMATION_STATES; id++) {
      expect(encodeState(decodeState(id))).toBe(id);
    }
  });

  it('starts with all tests unobserved', () => {
    const s = initialInformationState(RiskProfile.Medium);
    for (const t of [Test.ECG, Test.Troponin, Test.DDimer, Test.CXR]) {
      expect(s.results[t]).toBe(TestResult.Unobserved);
    }
  });
});

describe('DiagnosticWorkupEnv', () => {
  it('samples valid diseases under normalized priors', () => {
    const env = new DiagnosticWorkupEnv();
    const rng = mulberry32(1);
    env.setRng(rng);
    const counts = { 0: 0, 1: 0, 2: 0, 3: 0 };
    for (let i = 0; i < 500; i++) {
      env.reset(RiskProfile.Medium);
      counts[env.getTrueDisease()] += 1;
    }
    expect(Object.values(counts).every((c) => c > 0)).toBe(true);
  });

  it('terminates on diagnosis with finite reward', () => {
    const env = new DiagnosticWorkupEnv();
    const rng = mulberry32(2);
    env.setRng(rng);
    env.reset(RiskProfile.Medium);
    const result = env.step(WorkupAction.DiagnoseBenign);
    expect(result.terminated).toBe(true);
    expect(Number.isFinite(result.reward)).toBe(true);
  });

  it('posterior sums to one after a test', () => {
    const env = new DiagnosticWorkupEnv();
    const rng = mulberry32(3);
    env.setRng(rng);
    env.reset(RiskProfile.High);
    env.step(WorkupAction.OrderDDimer);
    const post = env.posterior(env.getState());
    const sum = Object.values(post).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 5);
  });
});

describe('DiagnosticWorldModel', () => {
  it('stores terminal transitions with done flag', () => {
    const env = new DiagnosticWorkupEnv();
    const rng = mulberry32(4);
    env.setRng(rng);
    env.reset(RiskProfile.Medium);
    const s = env.encode(env.getState());
    const model = new DiagnosticWorldModel();
    const result = env.step(WorkupAction.DiagnoseACS);
    const sNext = env.encode(result.obs);
    model.update(s, WorkupAction.DiagnoseACS, result.reward, sNext, true);
    const sample = model.sample(mulberry32(5));
    expect(sample?.done).toBe(true);
  });
});

describe('diagnostic training', () => {
  it('runs deterministically for fixed seed', () => {
    const a = runDiagnosticSingle({ episodes: 20, dreamUpdatesPerStep: 5 }, 99);
    const b = runDiagnosticSingle({ episodes: 20, dreamUpdatesPerStep: 5 }, 99);
    expect(a.history.length).toBe(20);
    expect(a.history.map((h) => h.episodeReturn)).toEqual(
      b.history.map((h) => h.episodeReturn)
    );
  });
});

describe('offline experiments', () => {
  it('produces finite reproducible checksum', () => {
    const lightProfile = {
      ...OFFLINE_DIAGNOSTIC_PROFILE,
      episodes: 30,
      seeds: 3,
      kValues: [0, 20],
      biasLevels: [0, 0.5],
      pePrevalenceValues: [1, 2],
      costMultipliers: [1, 2],
    };
    const r1 = runDiagnosticOfflineExperiments(lightProfile);
    const r2 = runDiagnosticOfflineExperiments(lightProfile);
    expect(diagnosticResultsChecksum(r1)).toBe(diagnosticResultsChecksum(r2));
    expect(r1.finalMetricsByAgent.length).toBeGreaterThanOrEqual(4);
  });
});

describe('config', () => {
  it('has sensitivities in [0,1]', () => {
    for (const d of [Disease.ACS, Disease.PE, Disease.Pneumonia, Disease.Benign]) {
      for (const t of [Test.ECG, Test.Troponin, Test.DDimer, Test.CXR]) {
        const p = DEFAULT_DIAGNOSTIC_CONFIG.sensitivity[d][t];
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    }
  });
});
