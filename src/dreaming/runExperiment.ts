import { createQ, epsilonGreedy, qUpdate, maxQ, type QTable } from './agents';
import { GridWorld, defaultWalls } from './env';
import {
  runTraining,
  TabularWorldModel,
  NoisyWorldModel,
  type EpisodeMetrics,
  type TrainingConfig,
  type TrainingResult,
} from './dyna';
import { mulberry32, type Rng } from './rng';

export interface MaxQGrid {
  grid: number[][];
  min: number;
  max: number;
}

export function snapshotMaxQGrid(Q: QTable, env: GridWorld): MaxQGrid {
  const grid: number[][] = [];
  let min = Infinity;
  let max = -Infinity;
  for (let r = 0; r < env.size; r++) {
    const row: number[] = [];
    for (let c = 0; c < env.size; c++) {
      const p = { r, c };
      if (env.isWall(p)) {
        row.push(NaN);
      } else {
        const s = env.encode(p);
        const v = maxQ(Q, s);
        row.push(v);
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    grid.push(row);
  }
  if (!Number.isFinite(min)) {
    min = 0;
    max = 0;
  }
  return { grid, min, max };
}

export function rollingMean(values: number[], window: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}

export interface CurvePoint {
  x: number;
  meanReturn: number;
  stderr: number;
}

export interface AggregatedCurve {
  agent: string;
  points: CurvePoint[];
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function stderr(values: number[]): number {
  if (values.length <= 1) return 0;
  const m = mean(values);
  const v = values.reduce((acc, x) => acc + (x - m) ** 2, 0) / (values.length - 1);
  return Math.sqrt(v / values.length);
}

export function aggregateReturnCurves(
  runsByAgent: Map<string, EpisodeMetrics[][]>,
  xKey: 'realEnvSteps' | 'totalUpdates',
  window = 10
): AggregatedCurve[] {
  const curves: AggregatedCurve[] = [];

  for (const [agent, runs] of runsByAgent) {
    const maxLen = Math.max(...runs.map((h) => h.length));
    const points: CurvePoint[] = [];

    for (let ep = 0; ep < maxLen; ep++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const run of runs) {
        if (ep >= run.length) continue;
        const m = run[ep]!;
        xs.push(xKey === 'realEnvSteps' ? m.realEnvSteps : m.totalUpdates);
        const smoothed = rollingMean(
          run.slice(0, ep + 1).map((h) => h.episodeReturn),
          window
        );
        ys.push(smoothed[smoothed.length - 1]!);
      }
      if (xs.length === 0) continue;
      points.push({
        x: mean(xs),
        meanReturn: mean(ys),
        stderr: stderr(ys),
      });
    }
    if (points.length > 0 && points[0]!.x > 0) {
      points.unshift({ x: 0, meanReturn: 0, stderr: 0 });
    }
    curves.push({ agent, points });
  }

  return curves;
}

export interface OfflineResults {
  meta: {
    episodes: number;
    seeds: number;
    size: number;
    walls: { r: number; c: number }[];
    kValues: number[];
    noiseValues: number[];
  };
  returnVsRealSteps: AggregatedCurve[];
  returnVsTotalUpdates: AggregatedCurve[];
  finalReturnVsNoise: { noise: number; meanReturn: number; stderr: number }[];
}

export const INTERACTIVE_PROFILE = {
  episodes: 80,
  dreamUpdatesPerStep: 20,
  modelNoise: 0,
  seed: 42,
};

export const OFFLINE_PROFILE = {
  episodes: 200,
  seeds: 10,
  kValues: [0, 1, 5, 20, 100],
  noiseValues: [0, 0.05, 0.1, 0.2],
  noiseSweepK: 20,
};

export function runSingle(config: TrainingConfig, seed: number) {
  const rng = mulberry32(seed);
  const env = config.env ?? new GridWorld();
  return runTraining({ ...config, env }, rng);
}

export function runOfflineExperiments(): OfflineResults {
  const { episodes, seeds, kValues, noiseValues, noiseSweepK } = OFFLINE_PROFILE;
  const env = new GridWorld();
  const walls = defaultWalls(env.size);

  const runsByAgentReal = new Map<string, EpisodeMetrics[][]>();
  const runsByAgentTotal = new Map<string, EpisodeMetrics[][]>();

  for (const k of kValues) {
    const label = `k=${k}`;
    const runs: EpisodeMetrics[][] = [];
    for (let seed = 0; seed < seeds; seed++) {
      const result = runSingle(
        { episodes, dreamUpdatesPerStep: k, modelNoise: 0, env: new GridWorld({ walls }) },
        seed + 1
      );
      runs.push(result.history);
    }
    runsByAgentReal.set(label, runs);
    runsByAgentTotal.set(label, runs);
  }

  const finalReturnVsNoise: OfflineResults['finalReturnVsNoise'] = [];
  for (const noise of noiseValues) {
    const finals: number[] = [];
    for (let seed = 0; seed < seeds; seed++) {
      const result = runSingle(
        {
          episodes,
          dreamUpdatesPerStep: noiseSweepK,
          modelNoise: noise,
          env: new GridWorld({ walls }),
        },
        seed + 100
      );
      const lastWindow = result.history.slice(-10).map((h) => h.episodeReturn);
      finals.push(mean(lastWindow));
    }
    finalReturnVsNoise.push({
      noise,
      meanReturn: mean(finals),
      stderr: stderr(finals),
    });
  }

  return {
    meta: {
      episodes,
      seeds,
      size: env.size,
      walls,
      kValues,
      noiseValues,
    },
    returnVsRealSteps: aggregateReturnCurves(runsByAgentReal, 'realEnvSteps'),
    returnVsTotalUpdates: aggregateReturnCurves(runsByAgentTotal, 'totalUpdates'),
    finalReturnVsNoise,
  };
}

/** Run training with chunked episode batches for UI responsiveness. */
export async function runTrainingAsync(
  config: TrainingConfig,
  seed: number,
  episodesPerFrame: number,
  onProgress: (history: EpisodeMetrics[], Q: QTable, env: GridWorld, done: boolean) => void,
  shouldCancel?: () => boolean
): Promise<TrainingResult> {
  const rng = mulberry32(seed);
  const env = config.env ?? new GridWorld();
  env.setRng(rng);

  const alpha = config.alpha ?? 0.1;
  const gamma = config.gamma ?? 0.95;
  const epsilon = config.epsilon ?? 0.1;
  const maxSteps = config.maxStepsPerEpisode ?? 200;
  const dreamK = config.dreamUpdatesPerStep;
  const modelNoise = config.modelNoise ?? 0;

  const validStates = env.validStates().map((p) => env.encode(p));
  const Q = createQ(env.nStates(), 4);
  const model =
    modelNoise > 0
      ? new NoisyWorldModel(env, modelNoise, validStates)
      : new TabularWorldModel();

  const history: EpisodeMetrics[] = [];
  let cumulativeRealSteps = 0;
  let cumulativeTotalUpdates = 0;
  let ep = 0;

  while (ep < config.episodes) {
    if (shouldCancel?.()) break;

    const batchEnd = Math.min(ep + episodesPerFrame, config.episodes);
    for (; ep < batchEnd; ep++) {
      env.reset();
      let s = env.encode(env.getPosition());
      let done = false;
      let steps = 0;
      let episodeReturn = 0;
      let realSteps = 0;
      let imagined = 0;
      let success = false;

      while (!done && steps < maxSteps) {
        const a = epsilonGreedy(Q, s, epsilon, rng);
        const { obs, reward, terminated } = env.step(a);
        const sNext = env.encode(obs);
        done = terminated;

        qUpdate(Q, s, a, reward, sNext, alpha, gamma, done);
        model.update(s, a, reward, sNext);
        realSteps += 1;
        cumulativeTotalUpdates += 1;
        episodeReturn += reward;

        for (let k = 0; k < dreamK; k++) {
          const t = model.sample(rng);
          if (!t) continue;
          qUpdate(Q, t.s, t.a, t.r, t.sNext, alpha, gamma, false);
          imagined += 1;
          cumulativeTotalUpdates += 1;
        }

        s = sNext;
        steps += 1;
        if (done) success = true;
      }

      cumulativeRealSteps += realSteps;
      history.push({
        episode: ep,
        episodeReturn,
        stepsToGoal: steps,
        realEnvSteps: cumulativeRealSteps,
        imaginedUpdates: imagined,
        totalUpdates: cumulativeTotalUpdates,
        success,
      });
    }

    onProgress(history, Q, env, ep >= config.episodes);
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
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

export { mulberry32, type Rng };
