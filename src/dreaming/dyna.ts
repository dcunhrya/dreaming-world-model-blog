import {
  DEFAULT_ALPHA,
  DEFAULT_EPSILON,
  DEFAULT_GAMMA,
  createQ,
  epsilonGreedy,
  qUpdate,
  type QTable,
} from './agents';
import { GridWorld, NUM_ACTIONS, type Action } from './env';
import { choice, type Rng } from './rng';

export interface TransitionSample {
  s: number;
  a: number;
  r: number;
  sNext: number;
}

export class TabularWorldModel {
  private transitionCounts = new Map<string, Map<number, number>>();
  private rewardSum = new Map<string, number>();
  private rewardCount = new Map<string, number>();
  private observedPairs: string[] = [];

  private pairKey(s: number, a: number): string {
    return `${s},${a}`;
  }

  update(s: number, a: number, r: number, sNext: number): void {
    const key = this.pairKey(s, a);
    if (!this.transitionCounts.has(key)) {
      this.transitionCounts.set(key, new Map());
      this.observedPairs.push(key);
    }
    const counts = this.transitionCounts.get(key)!;
    counts.set(sNext, (counts.get(sNext) ?? 0) + 1);
    this.rewardSum.set(key, (this.rewardSum.get(key) ?? 0) + r);
    this.rewardCount.set(key, (this.rewardCount.get(key) ?? 0) + 1);
  }

  sample(rng: Rng): TransitionSample | null {
    if (this.observedPairs.length === 0) return null;
    const key = choice(rng, this.observedPairs);
    const [sStr, aStr] = key.split(',');
    const s = Number(sStr);
    const a = Number(aStr);
    const counts = this.transitionCounts.get(key)!;
    const entries = [...counts.entries()];
    const total = entries.reduce((sum, [, c]) => sum + c, 0);
    let roll = rng() * total;
    let sNext = entries[0]![0];
    for (const [next, count] of entries) {
      roll -= count;
      if (roll <= 0) {
        sNext = next;
        break;
      }
    }
    const r =
      (this.rewardSum.get(key) ?? 0) / Math.max(1, this.rewardCount.get(key) ?? 1);
    return { s, a, r, sNext };
  }
}

export class NoisyWorldModel extends TabularWorldModel {
  constructor(
    private env: GridWorld,
    private modelNoise: number,
    private validStateIds: number[]
  ) {
    super();
  }

  override sample(rng: Rng): TransitionSample | null {
    const base = super.sample(rng);
    if (!base) return null;
    if (this.modelNoise > 0 && rng() < this.modelNoise) {
      base.sNext = choice(rng, this.validStateIds);
    }
    return base;
  }
}

export interface EpisodeMetrics {
  episode: number;
  episodeReturn: number;
  stepsToGoal: number;
  realEnvSteps: number;
  imaginedUpdates: number;
  totalUpdates: number;
  success: boolean;
}

export interface TrainingConfig {
  episodes: number;
  dreamUpdatesPerStep: number;
  modelNoise?: number;
  alpha?: number;
  gamma?: number;
  epsilon?: number;
  maxStepsPerEpisode?: number;
  seed?: number;
  env?: GridWorld;
  onEpisodeComplete?: (metrics: EpisodeMetrics, Q: QTable) => void;
}

export interface TrainingResult {
  history: EpisodeMetrics[];
  finalQ: QTable;
  envMeta: {
    size: number;
    walls: { r: number; c: number }[];
    start: { r: number; c: number };
    goal: { r: number; c: number };
  };
  cumulativeRealSteps: number;
  cumulativeTotalUpdates: number;
}

function makeModel(
  env: GridWorld,
  modelNoise: number,
  validStateIds: number[]
): TabularWorldModel {
  if (modelNoise > 0) {
    return new NoisyWorldModel(env, modelNoise, validStateIds);
  }
  return new TabularWorldModel();
}

export function runTraining(config: TrainingConfig, rng: Rng): TrainingResult {
  const env = config.env ?? new GridWorld();
  env.setRng(rng);

  const alpha = config.alpha ?? DEFAULT_ALPHA;
  const gamma = config.gamma ?? DEFAULT_GAMMA;
  const epsilon = config.epsilon ?? DEFAULT_EPSILON;
  const maxSteps = config.maxStepsPerEpisode ?? 200;
  const dreamK = config.dreamUpdatesPerStep;
  const modelNoise = config.modelNoise ?? 0;

  const validStates = env.validStates().map((p) => env.encode(p));
  const Q = createQ(env.nStates(), NUM_ACTIONS);
  const model = makeModel(env, modelNoise, validStates);

  const history: EpisodeMetrics[] = [];
  let cumulativeRealSteps = 0;
  let cumulativeTotalUpdates = 0;

  for (let ep = 0; ep < config.episodes; ep++) {
    const startPos = env.reset();
    let s = env.encode(startPos);
    let done = false;
    let steps = 0;
    let episodeReturn = 0;
    let realSteps = 0;
    let imagined = 0;
    let totalUpdates = 0;
    let success = false;

    while (!done && steps < maxSteps) {
      const a = epsilonGreedy(Q, s, epsilon, rng);
      const { obs, reward, terminated } = env.step(a);
      const sNext = env.encode(obs);
      done = terminated;

      qUpdate(Q, s, a, reward, sNext, alpha, gamma, done);
      model.update(s, a, reward, sNext);
      realSteps += 1;
      totalUpdates += 1;
      episodeReturn += reward;

      for (let k = 0; k < dreamK; k++) {
        const imaginedTransition = model.sample(rng);
        if (!imaginedTransition) continue;
        const { s: sHat, a: aHat, r: rHat, sNext: sNextHat } = imaginedTransition;
        qUpdate(Q, sHat, aHat, rHat, sNextHat, alpha, gamma, false);
        imagined += 1;
        totalUpdates += 1;
      }

      s = sNext;
      steps += 1;
      if (done) success = true;
    }

    cumulativeRealSteps += realSteps;
    cumulativeTotalUpdates += totalUpdates;

    const metrics: EpisodeMetrics = {
      episode: ep,
      episodeReturn,
      stepsToGoal: steps,
      realEnvSteps: cumulativeRealSteps,
      imaginedUpdates: imagined,
      totalUpdates: cumulativeTotalUpdates,
      success,
    };
    history.push(metrics);
    config.onEpisodeComplete?.(metrics, Q);
  }

  return {
    history,
    finalQ: Q,
    envMeta: {
      size: env.size,
      walls: [...env.walls].map((k) => {
        const [r, c] = k.split(',').map(Number);
        return { r: r!, c: c! };
      }),
      start: env.start,
      goal: env.goal,
    },
    cumulativeRealSteps,
    cumulativeTotalUpdates,
  };
}
