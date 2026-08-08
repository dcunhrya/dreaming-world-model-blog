import {
  DEFAULT_ALPHA,
  DEFAULT_EPSILON,
  DEFAULT_GAMMA,
  createQ,
  epsilonGreedy,
  qUpdate,
  type QTable,
} from '../agents';
import { mulberry32, type Rng } from '../rng';
import { mergeDiagnosticConfig } from './config';
import { DiagnosticWorkupEnv } from './env';
import { makeDiagnosticModel } from './worldModel';
import {
  Disease,
  NUM_WORKUP_ACTIONS,
  RiskProfile,
  WorkupAction,
  type DiagnosticConfig,
  type DiagnosticEpisodeMetrics,
  type DiagnosticTrainingConfig,
  type DiagnosticTrainingResult,
} from './types';

export function runDiagnosticTraining(
  userConfig: DiagnosticTrainingConfig,
  rng: Rng
): DiagnosticTrainingResult {
  const config = mergeDiagnosticConfig({
    pePrevalenceMultiplier: userConfig.pePrevalenceMultiplier ?? 1,
    modelBiasLevel: userConfig.modelBiasLevel ?? 0,
  });
  const env = new DiagnosticWorkupEnv(config);
  env.setRng(rng);

  const alpha = userConfig.alpha ?? DEFAULT_ALPHA;
  const gamma = userConfig.gamma ?? DEFAULT_GAMMA;
  const epsilon = userConfig.epsilon ?? DEFAULT_EPSILON;
  const dreamK = userConfig.dreamUpdatesPerStep;
  const risk = userConfig.riskProfile ?? RiskProfile.Medium;

  const Q = createQ(env.nStates(), NUM_WORKUP_ACTIONS);
  const model = makeDiagnosticModel(env, config);

  const history: DiagnosticEpisodeMetrics[] = [];
  let cumulativeRealSteps = 0;
  let cumulativeTotalUpdates = 0;

  for (let ep = 0; ep < userConfig.episodes; ep++) {
    const obs = env.reset(risk);
    let s = env.encode(obs);
    let done = false;
    let episodeReturn = 0;
    let realSteps = 0;
    let imagined = 0;
    let testsOrdered = 0;
    let stepsInEpisode = 0;
    let predictedDisease: Disease | null = null;
    let trueDisease = env.getTrueDisease();

    while (!done) {
      const action = epsilonGreedy(Q, s, epsilon, rng, NUM_WORKUP_ACTIONS) as WorkupAction;
      const result = env.step(action);
      const sNext = env.encode(result.obs);
      done = result.terminated || result.truncated;

      qUpdate(Q, s, action, result.reward, sNext, alpha, gamma, done);
      model.update(s, action, result.reward, sNext, done);
      realSteps += 1;
      cumulativeTotalUpdates += 1;
      episodeReturn += result.reward;
      stepsInEpisode += 1;

      if (result.info.testOrdered != null && result.info.testFinding != null) {
        testsOrdered += 1;
      }
      if (result.info.diagnosis != null) {
        predictedDisease = result.info.diagnosis;
      }

      for (let k = 0; k < dreamK; k++) {
        const imaginedTransition = model.sample(rng);
        if (!imaginedTransition) continue;
        qUpdate(
          Q,
          imaginedTransition.s,
          imaginedTransition.a,
          imaginedTransition.r,
          imaginedTransition.sNext,
          alpha,
          gamma,
          imaginedTransition.done
        );
        imagined += 1;
        cumulativeTotalUpdates += 1;
      }

      s = sNext;
      trueDisease = env.getTrueDisease();
    }

    cumulativeRealSteps += realSteps;
    const correct = predictedDisease === trueDisease;
    const missedCritical =
      !correct &&
      (trueDisease === Disease.ACS || trueDisease === Disease.PE);

    const metrics: DiagnosticEpisodeMetrics = {
      episode: ep,
      episodeReturn,
      realEnvSteps: realSteps,
      imaginedUpdates: imagined,
      totalUpdates: cumulativeTotalUpdates,
      cumulativeRealSteps,
      cumulativeTotalUpdates,
      correct,
      trueDisease,
      predictedDisease,
      testsOrdered,
      stepsInEpisode,
      missedCritical,
    };
    history.push(metrics);
    userConfig.onEpisodeComplete?.(metrics);
  }

  return { history, finalQ: Q, config };
}

export function runDiagnosticSingle(
  config: DiagnosticTrainingConfig,
  seed: number
): DiagnosticTrainingResult {
  return runDiagnosticTraining(config, mulberry32(seed));
}

export function greedyPolicy(Q: QTable, s: number): WorkupAction {
  return epsilonGreedy(Q, s, 0, () => 0, NUM_WORKUP_ACTIONS) as WorkupAction;
}
