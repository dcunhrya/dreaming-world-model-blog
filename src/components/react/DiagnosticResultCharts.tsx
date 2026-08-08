import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  ErrorBar,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import type { DiagnosticOfflineResults } from '../../dreaming/diagnostic/runExperiment';
import { agentColors, chartTooltipStyle, colors } from '../../lib/theme';
import MultiAgentLearningChart, { toMultiAgentCurves } from './MultiAgentLearningChart';

export type DiagnosticChartVariant =
  | 'returnVsRealSteps'
  | 'returnVsTotalUpdates'
  | 'accuracyVsRealSteps'
  | 'criticalRecallVsRealSteps'
  | 'biasFailure'
  | 'prevalenceShift'
  | 'costSafetyPareto'
  | 'finalMetrics';

const TITLES: Record<DiagnosticChartVariant, string> = {
  returnVsRealSteps: 'Sample efficiency: return vs real simulated patients',
  returnVsTotalUpdates: 'Compute efficiency: return vs total Q-updates',
  accuracyVsRealSteps: 'Diagnostic accuracy vs real simulated patients',
  criticalRecallVsRealSteps: 'ACS/PE recall vs real simulated patients',
  biasFailure: 'Model misspecification: return vs D-dimer bias (by dream budget)',
  prevalenceShift: 'Rare PE prevalence vs per-disease recall (Dyna-Q k=20)',
  costSafetyPareto: 'Cost–safety tradeoff: accuracy vs missed critical rate',
  finalMetrics: 'Final benchmark summary (mean return by agent)',
};

const DESCRIPTIONS: Record<DiagnosticChartVariant, string> = {
  returnVsRealSteps:
    'Higher dreaming budgets can improve returns with fewer real patient rollouts when the learned model is accurate.',
  returnVsTotalUpdates:
    'Once imagined updates are counted, heavy dreaming trades real interaction for compute.',
  accuracyVsRealSteps:
    'Fraction of episodes ending in the correct diagnosis, smoothed over training.',
  criticalRecallVsRealSteps:
    'Recall on ACS and PE only — the safety-sensitive subset of the synthetic cohort.',
  biasFailure:
    'Underestimating D-dimer sensitivity in the learned model hurts more as dreaming increases.',
  prevalenceShift:
    'When PE is rarer, aggregate accuracy can look fine while PE recall collapses if the model never sees enough real PE transitions.',
  costSafetyPareto:
    'Higher test costs push policies toward fewer orders; missed critical diagnoses track the safety side of the tradeoff.',
  finalMetrics:
    'Held-out-style summary after training; compare RL agents to prior-only and all-tests heuristics.',
};

const diagnosticAgentColors: Record<string, string> = {
  'Q-learning': colors.inkLight,
  'Dyna-Q k=1': '#C4A574',
  'Dyna-Q k=5': colors.cardinalLight,
  'Dyna-Q k=20': colors.cardinal,
  'Dyna-Q k=100': colors.cardinalDark,
  'Prior-only': '#94a3b8',
  'All-tests oracle': '#64748b',
};

export default function DiagnosticResultCharts({
  variant,
}: {
  variant: DiagnosticChartVariant;
}) {
  const [data, setData] = useState<DiagnosticOfflineResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.BASE_URL ?? '/';
    fetch(`${base}data/diagnostic_results.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const chartContent = useMemo(() => {
    if (!data) return null;

    if (variant === 'finalMetrics') {
      const rows = data.finalMetricsByAgent.map((a) => ({
        agent: a.agent,
        meanReturn: a.meanReturn,
        err: a.stderrReturn,
      }));
      return (
        <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 40, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            dataKey="agent"
            angle={-20}
            textAnchor="end"
            height={60}
            tick={{ fill: colors.inkMuted, fontSize: 10 }}
          />
          <YAxis tick={{ fill: colors.inkMuted, fontSize: 11 }} />
          <Tooltip contentStyle={chartTooltipStyle} />
          <Line
            type="monotone"
            dataKey="meanReturn"
            stroke={colors.cardinal}
            strokeWidth={2}
            dot={{ r: 4, fill: colors.cardinal }}
          >
            <ErrorBar dataKey="err" width={4} strokeWidth={1} stroke={colors.cardinalDark} />
          </Line>
        </LineChart>
      );
    }

    if (variant === 'biasFailure') {
      const agents = [...new Set(data.biasFailureCurves.flatMap((b) => b.points.map((p) => p.agent)))];
      const rows = data.biasFailureCurves.map((b) => {
        const row: Record<string, number | string> = { biasLevel: b.biasLevel };
        for (const p of b.points) row[p.agent] = p.meanReturn;
        return row;
      });
      return (
        <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            dataKey="biasLevel"
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            label={{ value: 'D-dimer model bias', position: 'insideBottom', offset: -8, fontSize: 11 }}
          />
          <YAxis tick={{ fill: colors.inkMuted, fontSize: 11 }} />
          <Tooltip contentStyle={chartTooltipStyle} />
          {agents.map((agent) => (
            <Line
              key={agent}
              type="monotone"
              dataKey={agent}
              name={agent}
              stroke={diagnosticAgentColors[agent] ?? colors.inkLight}
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          ))}
        </LineChart>
      );
    }

    if (variant === 'prevalenceShift') {
      const rows = data.prevalenceShift.map((p) => ({
        peMultiplier: p.peMultiplier,
        ACS: p.perDiseaseRecall.ACS ?? 0,
        PE: p.perDiseaseRecall.PE ?? 0,
        Pneumonia: p.perDiseaseRecall.Pneumonia ?? 0,
        Benign: p.perDiseaseRecall.Benign ?? 0,
      }));
      return (
        <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            dataKey="peMultiplier"
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            label={{ value: 'PE prevalence multiplier', position: 'insideBottom', offset: -8, fontSize: 11 }}
          />
          <YAxis domain={[0, 1]} tick={{ fill: colors.inkMuted, fontSize: 11 }} />
          <Tooltip contentStyle={chartTooltipStyle} />
          {['ACS', 'PE', 'Pneumonia', 'Benign'].map((d, i) => (
            <Line
              key={d}
              type="monotone"
              dataKey={d}
              name={d}
              stroke={[colors.cardinal, colors.cardinalDark, colors.cardinalLight, colors.inkLight][i]}
              strokeWidth={2}
              dot={{ r: 3 }}
            />
          ))}
        </LineChart>
      );
    }

    if (variant === 'costSafetyPareto') {
      return (
        <ScatterChart margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            type="number"
            dataKey="missedCriticalRate"
            name="Missed ACS/PE rate"
            domain={[0, 'auto']}
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            label={{ value: 'Missed critical rate', position: 'insideBottom', offset: -8, fontSize: 11 }}
          />
          <YAxis
            type="number"
            dataKey="accuracy"
            name="Accuracy"
            domain={[0, 1]}
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
          />
          <ZAxis type="number" dataKey="avgTests" range={[40, 200]} />
          <Tooltip contentStyle={chartTooltipStyle} cursor={{ strokeDasharray: '3 3' }} />
          <Scatter
            data={data.costSafetyPareto}
            fill={colors.cardinal}
            name="Policies"
          />
        </ScatterChart>
      );
    }

    const keyMap: Record<string, keyof DiagnosticOfflineResults> = {
      returnVsRealSteps: 'returnVsRealSteps',
      returnVsTotalUpdates: 'returnVsTotalUpdates',
      accuracyVsRealSteps: 'accuracyVsRealSteps',
      criticalRecallVsRealSteps: 'criticalRecallVsRealSteps',
    };
    const curveKey = keyMap[variant] ?? 'returnVsRealSteps';
    const curves = data[curveKey] as DiagnosticOfflineResults['returnVsRealSteps'];
    const xLabel = variant.includes('TotalUpdates') ? 'Total Q-updates' : 'Real simulated patients';

    return (
      <MultiAgentLearningChart
        curves={toMultiAgentCurves(curves)}
        xLabel={xLabel}
        agentStroke={(agent) =>
          diagnosticAgentColors[agent] ?? agentColors[agent] ?? colors.inkLight
        }
      />
    );
  }, [data, variant]);

  return (
    <div className="chart-card not-prose">
      <p className="mb-1 text-sm font-medium text-ink">{TITLES[variant]}</p>
      <p className="mb-4 text-sm text-ink-muted">{DESCRIPTIONS[variant]}</p>
      {error && <p className="text-sm text-accent">Failed to load results: {error}</p>}
      {!data && !error && (
        <p className="py-8 text-center text-sm text-ink-light">Loading experiment data…</p>
      )}
      {chartContent && (
        <div className="h-[300px] w-full">
          <ResponsiveContainer width="100%" height={300}>
            {chartContent}
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
