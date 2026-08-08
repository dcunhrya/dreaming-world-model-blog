import { argmaxQ, type QTable } from '../agents';
import { mulberry32, type Rng } from '../rng';
import { DiagnosticWorkupEnv, type DiagnosticStepResult } from './env';
import {
  ACTION_LABELS,
  DISEASE_LABELS,
  Disease,
  RiskProfile,
  WorkupAction,
  type InformationState,
} from './types';

export interface CaseStep {
  action: WorkupAction;
  actionLabel: string;
  reward: number;
  state: InformationState;
  posterior: Record<Disease, number>;
  info: DiagnosticStepResult['info'];
  terminated: boolean;
}

export interface CaseReplay {
  trueDisease: Disease;
  trueDiseaseLabel: string;
  steps: CaseStep[];
  totalReturn: number;
  correct: boolean;
}

export function replayLearnedCase(
  env: DiagnosticWorkupEnv,
  Q: QTable,
  seed: number,
  risk: RiskProfile = RiskProfile.Medium
): CaseReplay {
  const rng = mulberry32(seed);
  env.setRng(rng);
  env.reset(risk);
  let s = env.encode(env.getState());
  const steps: CaseStep[] = [];
  let totalReturn = 0;
  let done = false;

  while (!done) {
    const action = argmaxQ(Q, s) as WorkupAction;
    const result = env.step(action);
    totalReturn += result.reward;
    steps.push({
      action,
      actionLabel: ACTION_LABELS[action],
      reward: result.reward,
      state: env.getState(),
      posterior: env.posterior(env.getState()),
      info: result.info,
      terminated: result.terminated || result.truncated,
    });
    s = env.encode(result.obs);
    done = result.terminated || result.truncated;
  }

  const trueDisease = env.getTrueDisease();
  const last = steps[steps.length - 1];
  return {
    trueDisease,
    trueDiseaseLabel: DISEASE_LABELS[trueDisease],
    steps,
    totalReturn,
    correct: last?.info.diagnosis === trueDisease,
  };
}

export async function trainDiagnosticAsync(
  config: {
    episodes: number;
    dreamUpdatesPerStep: number;
    modelBiasLevel: number;
    pePrevalenceMultiplier: number;
    riskProfile: RiskProfile;
  },
  seed: number,
  onProgress?: (episode: number, total: number) => void
): Promise<QTable> {
  const { runDiagnosticTraining } = await import('./training');
  let lastEp = 0;
  const result = runDiagnosticTraining(
    {
      ...config,
      onEpisodeComplete: (m) => {
        if (m.episode - lastEp >= 4) {
          lastEp = m.episode;
          onProgress?.(m.episode + 1, config.episodes);
        }
      },
    },
    mulberry32(seed)
  );
  onProgress?.(config.episodes, config.episodes);
  return result.finalQ;
}
