import {
  Disease,
  Test,
  type DiagnosticConfig,
} from './types';

/**
 * Synthetic ED chest-pain simulator parameters.
 *
 * NOT derived from EHR data. Sensitivity values are simplified approximations
 * inspired by common clinical teaching ranges (e.g., troponin for ACS,
 * D-dimer for PE, infiltrate on CXR for pneumonia). Used for educational
 * demonstration only — not clinically validated.
 */
export const DEFAULT_DIAGNOSTIC_CONFIG: DiagnosticConfig = {
  basePrevalence: {
    [Disease.ACS]: 0.15,
    [Disease.PE]: 0.05,
    [Disease.Pneumonia]: 0.25,
    [Disease.Benign]: 0.55,
  },
  pePrevalenceMultiplier: 1.0,
  sensitivity: {
    [Disease.ACS]: {
      [Test.ECG]: 0.72,
      [Test.Troponin]: 0.88,
      [Test.DDimer]: 0.18,
      [Test.CXR]: 0.06,
    },
    [Disease.PE]: {
      [Test.ECG]: 0.12,
      [Test.Troponin]: 0.06,
      [Test.DDimer]: 0.92,
      [Test.CXR]: 0.08,
    },
    [Disease.Pneumonia]: {
      [Test.ECG]: 0.08,
      [Test.Troponin]: 0.05,
      [Test.DDimer]: 0.22,
      [Test.CXR]: 0.86,
    },
    [Disease.Benign]: {
      [Test.ECG]: 0.06,
      [Test.Troponin]: 0.03,
      [Test.DDimer]: 0.12,
      [Test.CXR]: 0.06,
    },
  },
  testCosts: {
    [Test.ECG]: 0.5,
    [Test.Troponin]: 1.0,
    [Test.DDimer]: 1.5,
    [Test.CXR]: 2.0,
  },
  correctDiagnosisReward: 10,
  wrongDiagnosisPenalty: 5,
  missedCriticalPenalty: 15,
  delayPenalty: 0.2,
  duplicateTestPenalty: 1.0,
  maxStepsPerEpisode: 12,
  modelBiasLevel: 0,
  biasedTest: Test.DDimer,
};

export function mergeDiagnosticConfig(
  overrides: Partial<DiagnosticConfig> = {}
): DiagnosticConfig {
  return { ...DEFAULT_DIAGNOSTIC_CONFIG, ...overrides };
}

/** Risk-profile multipliers applied to disease logits before normalization. */
export const RISK_PRIOR_MULTIPLIERS: Record<
  import('./types').RiskProfile,
  Record<Disease, number>
> = {
  0: { [Disease.ACS]: 0.5, [Disease.PE]: 0.4, [Disease.Pneumonia]: 0.9, [Disease.Benign]: 1.4 },
  1: { [Disease.ACS]: 1.0, [Disease.PE]: 1.0, [Disease.Pneumonia]: 1.0, [Disease.Benign]: 1.0 },
  2: { [Disease.ACS]: 1.8, [Disease.PE]: 1.6, [Disease.Pneumonia]: 1.1, [Disease.Benign]: 0.6 },
};

export const INTERACTIVE_DIAGNOSTIC_PROFILE = {
  episodes: 120,
  dreamUpdatesPerStep: 20,
  modelBiasLevel: 0,
  seed: 42,
};

export const OFFLINE_DIAGNOSTIC_PROFILE = {
  episodes: 150,
  seeds: 8,
  kValues: [0, 1, 5, 20, 100],
  biasLevels: [0, 0.25, 0.5, 0.75],
  biasSweepK: 20,
  pePrevalenceValues: [0.25, 0.5, 1.0, 2.0, 4.0],
  costMultipliers: [0.5, 1.0, 2.0, 4.0],
};
