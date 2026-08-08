import { mergeDiagnosticConfig, OFFLINE_DIAGNOSTIC_PROFILE } from './config';
import { DiagnosticWorkupEnv } from './env';
import { runDiagnosticSingle } from './training';
import {
  evaluateLearnedPolicy,
  runAllTestsBaseline,
  runPriorOnlyBaseline,
} from './baselines';
import { mulberry32 } from '../rng';
import { RiskProfile, Test, type DiagnosticEpisodeMetrics } from './types';

export interface DiagnosticCurvePoint {
  x: number;
  mean: number;
  stderr: number;
}

export interface DiagnosticAggregatedCurve {
  agent: string;
  metric: string;
  points: DiagnosticCurvePoint[];
}

export interface DiagnosticOfflineResults {
  meta: {
    episodes: number;
    seeds: number;
    kValues: number[];
    biasLevels: number[];
    pePrevalenceValues: number[];
    costMultipliers: number[];
    config: ReturnType<typeof mergeDiagnosticConfig>;
  };
  returnVsRealSteps: DiagnosticAggregatedCurve[];
  returnVsTotalUpdates: DiagnosticAggregatedCurve[];
  accuracyVsRealSteps: DiagnosticAggregatedCurve[];
  criticalRecallVsRealSteps: DiagnosticAggregatedCurve[];
  finalMetricsByAgent: {
    agent: string;
    meanReturn: number;
    stderrReturn: number;
    accuracy: number;
    macroRecall: number;
    criticalRecall: number;
    avgTests: number;
    missedCriticalRate: number;
  }[];
  biasFailureCurves: {
    biasLevel: number;
    points: { agent: string; meanReturn: number; stderr: number; criticalRecall: number }[];
  }[];
  prevalenceShift: {
    peMultiplier: number;
    perDiseaseRecall: Record<string, number>;
    accuracy: number;
    criticalRecall: number;
  }[];
  costSafetyPareto: {
    costMultiplier: number;
    agent: string;
    accuracy: number;
    missedCriticalRate: number;
    avgTests: number;
  }[];
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

function rollingMean(values: number[], window: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = values.slice(start, i + 1);
    out.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return out;
}

function aggregateMetricCurves(
  runsByAgent: Map<string, DiagnosticEpisodeMetrics[][]>,
  xKey: 'cumulativeRealSteps' | 'cumulativeTotalUpdates',
  yKey: keyof DiagnosticEpisodeMetrics,
  metricLabel: string,
  window = 8
): DiagnosticAggregatedCurve[] {
  const curves: DiagnosticAggregatedCurve[] = [];
  for (const [agent, runs] of runsByAgent) {
    const maxLen = Math.max(...runs.map((h) => h.length));
    const points: DiagnosticCurvePoint[] = [];
    for (let ep = 0; ep < maxLen; ep++) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const run of runs) {
        if (ep >= run.length) continue;
        const m = run[ep]!;
        xs.push(m[xKey] as number);
        const hist = run.slice(0, ep + 1).map((h) => Number(h[yKey]));
        const smoothed = rollingMean(hist, window);
        ys.push(smoothed[smoothed.length - 1]!);
      }
      if (xs.length === 0) continue;
      points.push({ x: mean(xs), mean: mean(ys), stderr: stderr(ys) });
    }
    if (points.length > 0 && points[0]!.x > 0) {
      points.unshift({ x: 0, mean: 0, stderr: 0 });
    }
    curves.push({ agent, metric: metricLabel, points });
  }
  return curves;
}

function summarizeRuns(
  agent: string,
  runs: DiagnosticEpisodeMetrics[][]
): DiagnosticOfflineResults['finalMetricsByAgent'][number] {
  const finals = runs.map((run) => {
    const tail = run.slice(-15);
    return {
      ret: mean(tail.map((h) => h.episodeReturn)),
      acc: mean(tail.map((h) => (h.correct ? 1 : 0))),
      macro: mean(tail.map((h) => (h.correct ? 1 : 0))),
      critical: mean(
        tail
          .filter((h) => h.trueDisease === 0 || h.trueDisease === 1)
          .map((h) => (h.correct ? 1 : 0))
      ),
      tests: mean(tail.map((h) => h.testsOrdered)),
      missed: mean(tail.map((h) => (h.missedCritical ? 1 : 0))),
    };
  });
  return {
    agent,
    meanReturn: mean(finals.map((f) => f.ret)),
    stderrReturn: stderr(finals.map((f) => f.ret)),
    accuracy: mean(finals.map((f) => f.acc)),
    macroRecall: mean(finals.map((f) => f.macro)),
    criticalRecall: mean(finals.map((f) => f.critical)) || 0,
    avgTests: mean(finals.map((f) => f.tests)),
    missedCriticalRate: mean(finals.map((f) => f.missed)),
  };
}

export function runDiagnosticOfflineExperiments(
  profile: typeof OFFLINE_DIAGNOSTIC_PROFILE = OFFLINE_DIAGNOSTIC_PROFILE
): DiagnosticOfflineResults {
  const { episodes, seeds, kValues, biasLevels, pePrevalenceValues, costMultipliers } =
    profile;
  const baseConfig = mergeDiagnosticConfig();

  const runsByAgentReal = new Map<string, DiagnosticEpisodeMetrics[][]>();
  const runsByAgentTotal = new Map<string, DiagnosticEpisodeMetrics[][]>();

  for (const k of kValues) {
    const label = k === 0 ? 'Q-learning' : `Dyna-Q k=${k}`;
    const runs: DiagnosticEpisodeMetrics[][] = [];
    for (let seed = 0; seed < seeds; seed++) {
      const result = runDiagnosticSingle(
        { episodes, dreamUpdatesPerStep: k, modelBiasLevel: 0, pePrevalenceMultiplier: 1 },
        seed + 1
      );
      runs.push(result.history);
    }
    runsByAgentReal.set(label, runs);
    runsByAgentTotal.set(label, runs);
  }

  const returnVsRealSteps = aggregateMetricCurves(
    runsByAgentReal,
    'cumulativeRealSteps',
    'episodeReturn',
    'return'
  );
  const returnVsTotalUpdates = aggregateMetricCurves(
    runsByAgentTotal,
    'cumulativeTotalUpdates',
    'episodeReturn',
    'return'
  );
  const accuracyVsRealSteps = aggregateMetricCurves(
    new Map(
      [...runsByAgentReal.entries()].map(([agent, runs]) => [
        agent,
        runs.map((run) =>
          run.map((m) => ({ ...m, correct: m.correct ? 1 : 0 }))
        ),
      ])
    ),
    'cumulativeRealSteps',
    'correct',
    'accuracy'
  );
  const criticalRecallVsRealSteps = aggregateMetricCurves(
    new Map(
      [...runsByAgentReal.entries()].map(([agent, runs]) => [
        agent,
        runs.map((run) =>
          run.map((m) => ({
            ...m,
            correct:
              (m.trueDisease === 0 || m.trueDisease === 1) && m.correct ? 1 : 0,
          }))
        ),
      ])
    ),
    'cumulativeRealSteps',
    'correct',
    'criticalRecall'
  );

  const finalMetricsByAgent: DiagnosticOfflineResults['finalMetricsByAgent'] = [];
  for (const [agent, runs] of runsByAgentReal) {
    finalMetricsByAgent.push(summarizeRuns(agent, runs));
  }

  const evalEnv = new DiagnosticWorkupEnv(baseConfig);
  const evalEpisodes = 80;
  const priorRuns = [];
  const allTestsRuns = [];
  for (let seed = 0; seed < seeds; seed++) {
    priorRuns.push(runPriorOnlyBaseline(evalEnv, evalEpisodes, seed + 500));
    allTestsRuns.push(runAllTestsBaseline(evalEnv, evalEpisodes, seed + 600));
  }
  finalMetricsByAgent.push({
    agent: 'Prior-only',
    meanReturn: mean(priorRuns.map((r) => mean(r.returns))),
    stderrReturn: stderr(priorRuns.map((r) => mean(r.returns))),
    accuracy: mean(priorRuns.map((r) => r.accuracy)),
    macroRecall: mean(priorRuns.map((r) => r.macroRecall)),
    criticalRecall: mean(priorRuns.map((r) => r.criticalRecall)),
    avgTests: mean(priorRuns.map((r) => r.avgTests)),
    missedCriticalRate: mean(priorRuns.map((r) => r.missedCriticalRate)),
  });
  finalMetricsByAgent.push({
    agent: 'All-tests oracle',
    meanReturn: mean(allTestsRuns.map((r) => mean(r.returns))),
    stderrReturn: stderr(allTestsRuns.map((r) => mean(r.returns))),
    accuracy: mean(allTestsRuns.map((r) => r.accuracy)),
    macroRecall: mean(allTestsRuns.map((r) => r.macroRecall)),
    criticalRecall: mean(allTestsRuns.map((r) => r.criticalRecall)),
    avgTests: mean(allTestsRuns.map((r) => r.avgTests)),
    missedCriticalRate: mean(allTestsRuns.map((r) => r.missedCriticalRate)),
  });

  const biasFailureCurves: DiagnosticOfflineResults['biasFailureCurves'] = [];
  for (const biasLevel of biasLevels) {
    const points: DiagnosticOfflineResults['biasFailureCurves'][number]['points'] = [];
    for (const k of [0, 5, 20, 100]) {
      const agent = k === 0 ? 'Q-learning' : `Dyna-Q k=${k}`;
      const returns: number[] = [];
      const critical: number[] = [];
      for (let seed = 0; seed < seeds; seed++) {
        const result = runDiagnosticSingle(
          { episodes, dreamUpdatesPerStep: k, modelBiasLevel: biasLevel },
          seed + 700 + biasLevel * 100 + k
        );
        const tail = result.history.slice(-15);
        returns.push(mean(tail.map((h) => h.episodeReturn)));
        critical.push(
          mean(
            tail
              .filter((h) => h.trueDisease === 0 || h.trueDisease === 1)
              .map((h) => (h.correct ? 1 : 0))
          ) || 0
        );
      }
      points.push({
        agent,
        meanReturn: mean(returns),
        stderr: stderr(returns),
        criticalRecall: mean(critical),
      });
    }
    biasFailureCurves.push({ biasLevel, points });
  }

  const prevalenceShift: DiagnosticOfflineResults['prevalenceShift'] = [];
  for (const peMultiplier of pePrevalenceValues) {
    const result = runDiagnosticSingle(
      { episodes, dreamUpdatesPerStep: 20, pePrevalenceMultiplier: peMultiplier },
      999
    );
    const tail = result.history.slice(-40);
    const byDisease: Record<string, { correct: number; total: number }> = {
      ACS: { correct: 0, total: 0 },
      PE: { correct: 0, total: 0 },
      Pneumonia: { correct: 0, total: 0 },
      Benign: { correct: 0, total: 0 },
    };
    const labels = ['ACS', 'PE', 'Pneumonia', 'Benign'];
    for (const m of tail) {
      const key = labels[m.trueDisease]!;
      byDisease[key]!.total += 1;
      if (m.correct) byDisease[key]!.correct += 1;
    }
    prevalenceShift.push({
      peMultiplier,
      perDiseaseRecall: Object.fromEntries(
        Object.entries(byDisease).map(([k, v]) => [k, v.total > 0 ? v.correct / v.total : 0])
      ),
      accuracy: mean(tail.map((h) => (h.correct ? 1 : 0))),
      criticalRecall:
        mean(
          tail
            .filter((h) => h.trueDisease === 0 || h.trueDisease === 1)
            .map((h) => (h.correct ? 1 : 0))
        ) || 0,
    });
  }

  const costSafetyPareto: DiagnosticOfflineResults['costSafetyPareto'] = [];
  for (const costMultiplier of costMultipliers) {
    const cfg = mergeDiagnosticConfig({
      testCosts: {
        [Test.ECG]: 0.5 * costMultiplier,
        [Test.Troponin]: 1.0 * costMultiplier,
        [Test.DDimer]: 1.5 * costMultiplier,
        [Test.CXR]: 2.0 * costMultiplier,
      },
    });
    const env = new DiagnosticWorkupEnv(cfg);
    for (const k of [0, 20]) {
      const train = runDiagnosticSingle({ episodes, dreamUpdatesPerStep: k }, 1200 + k);
      const evalResult = evaluateLearnedPolicy(
        env,
        train.finalQ,
        mulberry32(1300 + k),
        60,
        RiskProfile.Medium
      );
      costSafetyPareto.push({
        costMultiplier,
        agent: k === 0 ? 'Q-learning' : 'Dyna-Q k=20',
        accuracy: evalResult.accuracy,
        missedCriticalRate: evalResult.missedCriticalRate,
        avgTests: evalResult.avgTests,
      });
    }
  }

  const results: DiagnosticOfflineResults = {
    meta: {
      episodes,
      seeds,
      kValues,
      biasLevels,
      pePrevalenceValues,
      costMultipliers,
      config: baseConfig,
    },
    returnVsRealSteps,
    returnVsTotalUpdates,
    accuracyVsRealSteps,
    criticalRecallVsRealSteps,
    finalMetricsByAgent,
    biasFailureCurves,
    prevalenceShift,
    costSafetyPareto,
  };

  assertDiagnosticResults(results, profile);
  return results;
}

export function assertDiagnosticResults(
  results: DiagnosticOfflineResults,
  profile: typeof OFFLINE_DIAGNOSTIC_PROFILE = OFFLINE_DIAGNOSTIC_PROFILE
): void {
  if (results.returnVsRealSteps.length !== profile.kValues.length) {
    throw new Error('Unexpected number of return curves');
  }
  for (const curve of results.returnVsRealSteps) {
    if (curve.points.length === 0) throw new Error(`Empty curve for ${curve.agent}`);
    for (const p of curve.points) {
      if (!Number.isFinite(p.mean)) throw new Error(`Non-finite mean for ${curve.agent}`);
    }
  }
  if (results.finalMetricsByAgent.length < profile.kValues.length + 2) {
    throw new Error('Expected agent summaries for each k plus baselines');
  }
  if (results.biasFailureCurves.length !== profile.biasLevels.length) {
    throw new Error('Bias curve count mismatch');
  }
}

/** Deterministic checksum for reproducibility tests. */
export function diagnosticResultsChecksum(results: DiagnosticOfflineResults): number {
  let sum = 0;
  for (const row of results.finalMetricsByAgent) {
    sum += Math.round(row.meanReturn * 1000);
    sum += Math.round(row.accuracy * 1000);
  }
  return sum;
}
