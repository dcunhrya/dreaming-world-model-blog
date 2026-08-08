import {
  RiskProfile,
  Test,
  TestResult,
  RISK_LABELS,
  TEST_LABELS,
  RESULT_LABELS,
  type InformationState,
} from './types';

export const NUM_TESTS = 4;
export const NUM_RESULT_LEVELS = 3;
export const NUM_RISK_LEVELS = 3;

/** Total information states: 3 risk profiles × 3^4 test-result combinations = 243. */
export const NUM_INFORMATION_STATES = NUM_RISK_LEVELS * NUM_RESULT_LEVELS ** NUM_TESTS;

export function initialInformationState(risk: RiskProfile): InformationState {
  return {
    risk,
    results: {
      [Test.ECG]: TestResult.Unobserved,
      [Test.Troponin]: TestResult.Unobserved,
      [Test.DDimer]: TestResult.Unobserved,
      [Test.CXR]: TestResult.Unobserved,
    },
  };
}

export function encodeState(state: InformationState): number {
  let id = state.risk * NUM_RESULT_LEVELS ** NUM_TESTS;
  const tests = [Test.ECG, Test.Troponin, Test.DDimer, Test.CXR];
  for (let i = 0; i < tests.length; i++) {
    id += state.results[tests[i]!]! * NUM_RESULT_LEVELS ** (NUM_TESTS - 1 - i);
  }
  return id;
}

export function decodeState(id: number): InformationState {
  if (id < 0 || id >= NUM_INFORMATION_STATES) {
    throw new RangeError(`Invalid state id: ${id}`);
  }
  let rem = id;
  const risk = Math.floor(rem / NUM_RESULT_LEVELS ** NUM_TESTS) as RiskProfile;
  rem %= NUM_RESULT_LEVELS ** NUM_TESTS;
  const tests = [Test.ECG, Test.Troponin, Test.DDimer, Test.CXR];
  const results = {} as Record<Test, TestResult>;
  for (let i = 0; i < tests.length; i++) {
    const pow = NUM_RESULT_LEVELS ** (NUM_TESTS - 1 - i);
    const val = Math.floor(rem / pow);
    rem %= pow;
    results[tests[i]!] = val as TestResult;
  }
  return { risk, results };
}

export function actionToTest(action: number): Test | null {
  if (action >= 0 && action <= 3) return action as Test;
  return null;
}

export function actionToDisease(action: number): import('./types').Disease | null {
  if (action >= 4 && action <= 7) return (action - 4) as import('./types').Disease;
  return null;
}

export function isTerminalState(_state: InformationState): boolean {
  return false;
}

export function stateDescription(state: InformationState): string {
  const parts = [RISK_LABELS[state.risk]];
  for (const t of [Test.ECG, Test.Troponin, Test.DDimer, Test.CXR]) {
    const r = state.results[t];
    if (r !== TestResult.Unobserved) {
      parts.push(`${TEST_LABELS[t]}: ${RESULT_LABELS[r]}`);
    }
  }
  return parts.join(' · ');
}
