/** Hidden ground-truth diagnosis (not observed by the agent). */
export enum Disease {
  ACS = 0,
  PE = 1,
  Pneumonia = 2,
  Benign = 3,
}

export const DISEASE_LABELS: Record<Disease, string> = {
  [Disease.ACS]: 'ACS',
  [Disease.PE]: 'PE',
  [Disease.Pneumonia]: 'Pneumonia',
  [Disease.Benign]: 'Benign',
};

export const CRITICAL_DISEASES = [Disease.ACS, Disease.PE];

/** Ordered diagnostic tests available in the workup. */
export enum Test {
  ECG = 0,
  Troponin = 1,
  DDimer = 2,
  CXR = 3,
}

export const TEST_LABELS: Record<Test, string> = {
  [Test.ECG]: 'ECG',
  [Test.Troponin]: 'Troponin',
  [Test.DDimer]: 'D-dimer',
  [Test.CXR]: 'Chest X-ray',
};

/** Test result visibility in the information state. */
export enum TestResult {
  Unobserved = 0,
  Negative = 1,
  Positive = 2,
}

export const RESULT_LABELS: Record<TestResult, string> = {
  [TestResult.Unobserved]: '—',
  [TestResult.Negative]: 'Negative',
  [TestResult.Positive]: 'Positive',
};

/** Observed risk profile at presentation (affects disease priors). */
export enum RiskProfile {
  Low = 0,
  Medium = 1,
  High = 2,
}

export const RISK_LABELS: Record<RiskProfile, string> = {
  [RiskProfile.Low]: 'Low risk',
  [RiskProfile.Medium]: 'Medium risk',
  [RiskProfile.High]: 'High risk',
};

/** Actions: order one of four tests or commit to one of four diagnoses. */
export enum WorkupAction {
  OrderECG = 0,
  OrderTroponin = 1,
  OrderDDimer = 2,
  OrderCXR = 3,
  DiagnoseACS = 4,
  DiagnosePE = 5,
  DiagnosePneumonia = 6,
  DiagnoseBenign = 7,
}

export const NUM_WORKUP_ACTIONS = 8;

export const ACTION_LABELS: Record<WorkupAction, string> = {
  [WorkupAction.OrderECG]: 'Order ECG',
  [WorkupAction.OrderTroponin]: 'Order troponin',
  [WorkupAction.OrderDDimer]: 'Order D-dimer',
  [WorkupAction.OrderCXR]: 'Order chest X-ray',
  [WorkupAction.DiagnoseACS]: 'Diagnose ACS',
  [WorkupAction.DiagnosePE]: 'Diagnose PE',
  [WorkupAction.DiagnosePneumonia]: 'Diagnose pneumonia',
  [WorkupAction.DiagnoseBenign]: 'Diagnose benign cause',
};

export interface InformationState {
  risk: RiskProfile;
  results: Record<Test, TestResult>;
}

export interface DiagnosticConfig {
  /** Base disease prevalence before risk-profile adjustment. Must sum to 1. */
  basePrevalence: Record<Disease, number>;
  /** Multiplier applied to PE prevalence (for rare-disease sweeps). */
  pePrevalenceMultiplier: number;
  /** P(positive test | disease, test). Literature-informed toy values. */
  sensitivity: Record<Disease, Record<Test, number>>;
  /** Test costs subtracted from reward when ordered. */
  testCosts: Record<Test, number>;
  /** Reward for correct diagnosis. */
  correctDiagnosisReward: number;
  /** Penalty for incorrect diagnosis. */
  wrongDiagnosisPenalty: number;
  /** Extra penalty when true disease is ACS or PE and diagnosis is wrong. */
  missedCriticalPenalty: number;
  /** Per-step delay penalty (encourages efficient workups). */
  delayPenalty: number;
  /** Penalty for ordering a test that was already performed. */
  duplicateTestPenalty: number;
  /** Maximum steps per episode before truncation. */
  maxStepsPerEpisode: number;
  /** Fraction by which learned model sensitivity is reduced (0 = accurate, 1 = worst). */
  modelBiasLevel: number;
  /** Which test's sensitivity is biased in the learned model. */
  biasedTest: Test;
}

export interface DiagnosticEpisodeMetrics {
  episode: number;
  episodeReturn: number;
  realEnvSteps: number;
  imaginedUpdates: number;
  totalUpdates: number;
  cumulativeRealSteps: number;
  cumulativeTotalUpdates: number;
  correct: boolean;
  trueDisease: Disease;
  predictedDisease: Disease | null;
  testsOrdered: number;
  stepsInEpisode: number;
  missedCritical: boolean;
}

export interface DiagnosticTrainingConfig {
  episodes: number;
  dreamUpdatesPerStep: number;
  modelBiasLevel?: number;
  alpha?: number;
  gamma?: number;
  epsilon?: number;
  seed?: number;
  riskProfile?: RiskProfile;
  pePrevalenceMultiplier?: number;
  onEpisodeComplete?: (metrics: DiagnosticEpisodeMetrics) => void;
}

export interface DiagnosticTrainingResult {
  history: DiagnosticEpisodeMetrics[];
  finalQ: import('../agents').QTable;
  config: DiagnosticConfig;
}
