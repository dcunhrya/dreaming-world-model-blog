import { argmaxQ, type QTable } from '../agents';
import { mulberry32, type Rng } from '../rng';
import { DiagnosticWorkupEnv } from './env';
import { initialInformationState } from './state';
import {
  Disease,
  RiskProfile,
  Test,
  TestResult,
  WorkupAction,
} from './types';

/** Always diagnose the highest prior disease without ordering tests. */
export function priorOnlyAction(env: DiagnosticWorkupEnv, risk: RiskProfile): WorkupAction {
  const posterior = env.posterior(initialInformationState(risk));
  let best: Disease = Disease.Benign;
  let bestP = -1;
  for (const d of [Disease.ACS, Disease.PE, Disease.Pneumonia, Disease.Benign]) {
    if (posterior[d] > bestP) {
      bestP = posterior[d];
      best = d;
    }
  }
  return (WorkupAction.DiagnoseACS + best) as WorkupAction;
}

/** Order all unobserved tests, then diagnose via simulator posterior. */
export function allTestsThenPosteriorPolicy(
  env: DiagnosticWorkupEnv,
  stateId: number
): WorkupAction {
  const state = env.decode(stateId);
  for (const test of [Test.ECG, Test.Troponin, Test.DDimer, Test.CXR]) {
    if (state.results[test] === TestResult.Unobserved) {
      return test as WorkupAction;
    }
  }
  const posterior = env.posterior(state);
  let best: Disease = Disease.Benign;
  let bestP = -1;
  for (const d of [Disease.ACS, Disease.PE, Disease.Pneumonia, Disease.Benign]) {
    if (posterior[d] > bestP) {
      bestP = posterior[d];
      best = d;
    }
  }
  return (WorkupAction.DiagnoseACS + best) as WorkupAction;
}

export function evaluateFixedPolicy(
  env: DiagnosticWorkupEnv,
  rng: Rng,
  episodes: number,
  risk: RiskProfile,
  chooseAction: (stateId: number, step: number) => WorkupAction
): {
  returns: number[];
  accuracy: number;
  macroRecall: number;
  criticalRecall: number;
  avgTests: number;
  missedCriticalRate: number;
} {
  const returns: number[] = [];
  const confusion: Record<Disease, Record<Disease, number>> = {
    [Disease.ACS]: { [Disease.ACS]: 0, [Disease.PE]: 0, [Disease.Pneumonia]: 0, [Disease.Benign]: 0 },
    [Disease.PE]: { [Disease.ACS]: 0, [Disease.PE]: 0, [Disease.Pneumonia]: 0, [Disease.Benign]: 0 },
    [Disease.Pneumonia]: { [Disease.ACS]: 0, [Disease.PE]: 0, [Disease.Pneumonia]: 0, [Disease.Benign]: 0 },
    [Disease.Benign]: { [Disease.ACS]: 0, [Disease.PE]: 0, [Disease.Pneumonia]: 0, [Disease.Benign]: 0 },
  };
  let correct = 0;
  let testsTotal = 0;
  let missedCritical = 0;
  let criticalTotal = 0;
  let criticalCorrect = 0;

  for (let ep = 0; ep < episodes; ep++) {
    env.setRng(rng);
    const obs = env.reset(risk);
    let s = env.encode(obs);
    let done = false;
    let totalReturn = 0;
    let tests = 0;
    let predicted: Disease | null = null;
    let step = 0;

    while (!done) {
      const action = chooseAction(s, step);
      const result = env.step(action);
      totalReturn += result.reward;
      if (result.info.testOrdered != null && result.info.testFinding != null) tests += 1;
      if (result.info.diagnosis != null) predicted = result.info.diagnosis;
      s = env.encode(result.obs);
      done = result.terminated || result.truncated;
      step += 1;
    }

    const trueD = env.getTrueDisease();
    if (predicted != null) confusion[trueD][predicted] += 1;
    if (predicted === trueD) correct += 1;
    if (trueD === Disease.ACS || trueD === Disease.PE) {
      criticalTotal += 1;
      if (predicted === trueD) criticalCorrect += 1;
      if (predicted !== trueD) missedCritical += 1;
    }
    returns.push(totalReturn);
    testsTotal += tests;
  }

  const diseases = [Disease.ACS, Disease.PE, Disease.Pneumonia, Disease.Benign];
  let recallSum = 0;
  for (const d of diseases) {
    const row = confusion[d];
    const total = Object.values(row).reduce((a, b) => a + b, 0);
    recallSum += total > 0 ? row[d] / total : 0;
  }

  return {
    returns,
    accuracy: correct / episodes,
    macroRecall: recallSum / diseases.length,
    criticalRecall: criticalTotal > 0 ? criticalCorrect / criticalTotal : 0,
    avgTests: testsTotal / episodes,
    missedCriticalRate: criticalTotal > 0 ? missedCritical / criticalTotal : 0,
  };
}

export function evaluateLearnedPolicy(
  env: DiagnosticWorkupEnv,
  Q: QTable,
  rng: Rng,
  episodes: number,
  risk: RiskProfile
) {
  return evaluateFixedPolicy(env, rng, episodes, risk, (stateId) =>
    argmaxQ(Q, stateId) as WorkupAction
  );
}

export function runPriorOnlyBaseline(
  env: DiagnosticWorkupEnv,
  episodes: number,
  seed: number,
  risk: RiskProfile = RiskProfile.Medium
) {
  return evaluateFixedPolicy(env, mulberry32(seed), episodes, risk, () =>
    priorOnlyAction(env, risk)
  );
}

export function runAllTestsBaseline(
  env: DiagnosticWorkupEnv,
  episodes: number,
  seed: number,
  risk: RiskProfile = RiskProfile.Medium
) {
  return evaluateFixedPolicy(env, mulberry32(seed), episodes, risk, (stateId) =>
    allTestsThenPosteriorPolicy(env, stateId)
  );
}
