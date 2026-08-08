import type { Rng } from '../rng';
import { weightedChoice } from '../rng';
import { mergeDiagnosticConfig, RISK_PRIOR_MULTIPLIERS } from './config';
import {
  decodeState,
  encodeState,
  initialInformationState,
  NUM_INFORMATION_STATES,
} from './state';
import {
  Disease,
  NUM_WORKUP_ACTIONS,
  RiskProfile,
  Test,
  TestResult,
  WorkupAction,
  type DiagnosticConfig,
  type InformationState,
} from './types';

export interface DiagnosticStepResult {
  obs: InformationState;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: {
    trueDisease: Disease;
    testOrdered: Test | null;
    testFinding: TestResult | null;
    diagnosis: Disease | null;
  };
}

export class DiagnosticWorkupEnv {
  readonly config: DiagnosticConfig;
  private rng: Rng | null = null;
  private risk: RiskProfile = RiskProfile.Medium;
  private state: InformationState = initialInformationState(RiskProfile.Medium);
  private trueDisease: Disease = Disease.Benign;
  private steps = 0;

  constructor(config: Partial<DiagnosticConfig> = {}) {
    this.config = mergeDiagnosticConfig(config);
  }

  setRng(rng: Rng): void {
    this.rng = rng;
  }

  nStates(): number {
    return NUM_INFORMATION_STATES;
  }

  nActions(): number {
    return NUM_WORKUP_ACTIONS;
  }

  encode(state: InformationState): number {
    return encodeState(state);
  }

  decode(id: number): InformationState {
    return decodeState(id);
  }

  getTrueDisease(): Disease {
    return this.trueDisease;
  }

  getState(): InformationState {
    return {
      risk: this.state.risk,
      results: { ...this.state.results },
    };
  }

  /** Sample hidden disease from risk-adjusted priors. */
  sampleDisease(risk: RiskProfile, rng: Rng): Disease {
    const mult = RISK_PRIOR_MULTIPLIERS[risk];
    const weights: { item: Disease; weight: number }[] = [];
    let total = 0;
    for (const d of [Disease.ACS, Disease.PE, Disease.Pneumonia, Disease.Benign]) {
      let w = this.config.basePrevalence[d] * mult[d];
      if (d === Disease.PE) w *= this.config.pePrevalenceMultiplier;
      weights.push({ item: d, weight: w });
      total += w;
    }
    for (const w of weights) w.weight /= total;
    return weightedChoice(rng, weights);
  }

  /** Sample test finding from ground-truth sensitivity (positive rate). */
  sampleFinding(disease: Disease, test: Test, rng: Rng): TestResult {
    const pPos = this.config.sensitivity[disease][test];
    return rng() < pPos ? TestResult.Positive : TestResult.Negative;
  }

  reset(risk: RiskProfile = RiskProfile.Medium): InformationState {
    if (!this.rng) throw new Error('RNG not set');
    this.risk = risk;
    this.state = initialInformationState(risk);
    this.trueDisease = this.sampleDisease(risk, this.rng);
    this.steps = 0;
    return this.getState();
  }

  validActions(state: InformationState): WorkupAction[] {
    const actions: WorkupAction[] = [];
    for (let a = WorkupAction.OrderECG; a <= WorkupAction.OrderCXR; a++) {
      actions.push(a);
    }
    for (let a = WorkupAction.DiagnoseACS; a <= WorkupAction.DiagnoseBenign; a++) {
      actions.push(a);
    }
    return actions;
  }

  step(action: WorkupAction): DiagnosticStepResult {
    if (!this.rng) throw new Error('RNG not set');
    this.steps += 1;
    let reward = -this.config.delayPenalty;
    let terminated = false;
    let testOrdered: Test | null = null;
    let testFinding: TestResult | null = null;
    let diagnosis: Disease | null = null;

    if (action >= WorkupAction.OrderECG && action <= WorkupAction.OrderCXR) {
      const test = action as Test;
      testOrdered = test;
      const current = this.state.results[test];
      if (current !== TestResult.Unobserved) {
        reward -= this.config.duplicateTestPenalty;
      } else {
        testFinding = this.sampleFinding(this.trueDisease, test, this.rng);
        this.state.results[test] = testFinding;
        reward -= this.config.testCosts[test];
      }
    } else {
      diagnosis = (action - WorkupAction.DiagnoseACS) as Disease;
      terminated = true;
      if (diagnosis === this.trueDisease) {
        reward += this.config.correctDiagnosisReward;
      } else {
        reward -= this.config.wrongDiagnosisPenalty;
        if (
          (this.trueDisease === Disease.ACS || this.trueDisease === Disease.PE) &&
          diagnosis !== this.trueDisease
        ) {
          reward -= this.config.missedCriticalPenalty;
        }
      }
    }

    const truncated =
      !terminated && this.steps >= this.config.maxStepsPerEpisode;

    if (truncated) {
      reward -= this.config.wrongDiagnosisPenalty;
      if (this.trueDisease === Disease.ACS || this.trueDisease === Disease.PE) {
        reward -= this.config.missedCriticalPenalty;
      }
    }

    return {
      obs: this.getState(),
      reward,
      terminated: terminated || truncated,
      truncated,
      info: {
        trueDisease: this.trueDisease,
        testOrdered,
        testFinding,
        diagnosis,
      },
    };
  }

  /** Simulator-derived posterior P(disease | observed state) for explanation UI only. */
  posterior(state: InformationState): Record<Disease, number> {
    const mult = RISK_PRIOR_MULTIPLIERS[state.risk];
    const logPrior: Record<Disease, number> = {
      [Disease.ACS]: Math.log(this.config.basePrevalence[Disease.ACS] * mult[Disease.ACS]),
      [Disease.PE]: Math.log(
        this.config.basePrevalence[Disease.PE] *
          mult[Disease.PE] *
          this.config.pePrevalenceMultiplier
      ),
      [Disease.Pneumonia]: Math.log(
        this.config.basePrevalence[Disease.Pneumonia] * mult[Disease.Pneumonia]
      ),
      [Disease.Benign]: Math.log(
        this.config.basePrevalence[Disease.Benign] * mult[Disease.Benign]
      ),
    };

    for (const test of [Test.ECG, Test.Troponin, Test.DDimer, Test.CXR]) {
      const result = state.results[test];
      if (result === TestResult.Unobserved) continue;
      for (const d of [Disease.ACS, Disease.PE, Disease.Pneumonia, Disease.Benign]) {
        const pPos = this.config.sensitivity[d][test];
        const likelihood = result === TestResult.Positive ? pPos : 1 - pPos;
        logPrior[d] += Math.log(Math.max(likelihood, 1e-9));
      }
    }

    const maxLog = Math.max(...Object.values(logPrior));
    const unnorm = Object.fromEntries(
      Object.entries(logPrior).map(([k, v]) => [k, Math.exp(v - maxLog)])
    ) as Record<Disease, number>;
    const sum = Object.values(unnorm).reduce((a, b) => a + b, 0);
    return Object.fromEntries(
      Object.entries(unnorm).map(([k, v]) => [k, v / sum])
    ) as Record<Disease, number>;
  }
}

/** Replay a single patient case with a fixed policy for the interactive explorer. */
export function replayCase(
  env: DiagnosticWorkupEnv,
  rng: Rng,
  risk: RiskProfile,
  chooseAction: (stateId: number, valid: WorkupAction[]) => WorkupAction
): {
  trueDisease: Disease;
  steps: {
    action: WorkupAction;
    reward: number;
    state: InformationState;
    posterior: Record<Disease, number>;
    info: DiagnosticStepResult['info'];
  }[];
  totalReturn: number;
  correct: boolean;
} {
  env.setRng(rng);
  const obs = env.reset(risk);
  let s = env.encode(obs);
  const steps: {
    action: WorkupAction;
    reward: number;
    state: InformationState;
    posterior: Record<Disease, number>;
    info: DiagnosticStepResult['info'];
  }[] = [];
  let totalReturn = 0;
  let done = false;

  while (!done) {
    const valid = env.validActions(env.getState());
    const action = chooseAction(s, valid);
    const result = env.step(action);
    totalReturn += result.reward;
    steps.push({
      action,
      reward: result.reward,
      state: env.getState(),
      posterior: env.posterior(env.getState()),
      info: result.info,
    });
    s = env.encode(result.obs);
    done = result.terminated || result.truncated;
  }

  const last = steps[steps.length - 1];
  const correct = last?.info.diagnosis === env.getTrueDisease();

  return {
    trueDisease: env.getTrueDisease(),
    steps,
    totalReturn,
    correct: !!correct,
  };
}
