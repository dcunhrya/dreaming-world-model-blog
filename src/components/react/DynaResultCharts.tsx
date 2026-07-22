import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  ErrorBar,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { OfflineResults } from '../../dreaming/runExperiment';
import { agentColors, chartTooltipStyle, colors } from '../../lib/theme';

export type ChartVariant = 'returnVsRealSteps' | 'returnVsTotalUpdates' | 'returnVsNoise';

const TITLES: Record<ChartVariant, string> = {
  returnVsRealSteps: 'Sample efficiency: return vs real environment steps',
  returnVsTotalUpdates: 'Compute efficiency: return vs total Q-updates',
  returnVsNoise: 'Failure mode: final return vs model noise (k = 20)',
};

const DESCRIPTIONS: Record<ChartVariant, string> = {
  returnVsRealSteps:
    'Higher dreaming budgets reach good returns with fewer real interactions when the model is accurate.',
  returnVsTotalUpdates:
    'Once total compute is counted, heavy dreaming is less impressive — you traded real steps for imagined updates.',
  returnVsNoise:
    'Injected transition noise makes the world model lie; the agent optimizes the dream, not the real grid.',
};

function flattenCurves(
  results: OfflineResults,
  key: 'returnVsRealSteps' | 'returnVsTotalUpdates'
) {
  const rows: { x: number; [agent: string]: number | string | null }[] = [];
  const curves = results[key];
  const xSet = new Set<number>();
  for (const curve of curves) {
    for (const p of curve.points) xSet.add(Math.round(p.x));
  }
  const xs = [...xSet].sort((a, b) => a - b);
  for (const x of xs) {
    const row: { x: number; [agent: string]: number | string | null } = { x };
    for (const curve of curves) {
      const nearest = curve.points.reduce((best, p) =>
        Math.abs(p.x - x) < Math.abs(best.x - x) ? p : best
      );
      if (Math.abs(nearest.x - x) < nearest.x * 0.15 + 50) {
        row[curve.agent] = nearest.meanReturn;
        row[`${curve.agent}_err`] = nearest.stderr;
      }
    }
    rows.push(row);
  }
  return { rows, agents: curves.map((c) => c.agent) };
}

export default function DynaResultCharts({ variant }: { variant: ChartVariant }) {
  const [data, setData] = useState<OfflineResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const base = import.meta.env.BASE_URL ?? '/';
    fetch(`${base}data/dyna_results.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const chartContent = useMemo(() => {
    if (!data) return null;

    if (variant === 'returnVsNoise') {
      const noiseData = data.finalReturnVsNoise.map((d) => ({
        noise: d.noise,
        meanReturn: d.meanReturn,
        err: d.stderr,
      }));
      return (
        <LineChart data={noiseData} margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis
            dataKey="noise"
            tick={{ fill: colors.inkMuted, fontSize: 11 }}
            label={{
              value: 'Model noise',
              position: 'insideBottom',
              offset: -8,
              fontSize: 11,
              fill: colors.inkMuted,
            }}
          />
          <YAxis tick={{ fill: colors.inkMuted, fontSize: 11 }} />
          <Tooltip contentStyle={chartTooltipStyle} />
          <Line
            type="monotone"
            dataKey="meanReturn"
            name="Mean return"
            stroke={colors.cardinal}
            strokeWidth={2}
            dot={{ r: 4, fill: colors.cardinal }}
          >
            <ErrorBar dataKey="err" width={4} strokeWidth={1} stroke={colors.cardinalDark} />
          </Line>
        </LineChart>
      );
    }

    const key =
      variant === 'returnVsRealSteps' ? 'returnVsRealSteps' : 'returnVsTotalUpdates';
    const { rows, agents } = flattenCurves(data, key);
    const xLabel =
      variant === 'returnVsRealSteps' ? 'Real env steps' : 'Total Q-updates';

    return (
      <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 24, left: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
        <XAxis
          dataKey="x"
          tick={{ fill: colors.inkMuted, fontSize: 11 }}
          label={{
            value: xLabel,
            position: 'insideBottom',
            offset: -8,
            fontSize: 11,
            fill: colors.inkMuted,
          }}
        />
        <YAxis tick={{ fill: colors.inkMuted, fontSize: 11 }} />
        <Tooltip contentStyle={chartTooltipStyle} />
        {agents.map((agent) => (
          <Line
            key={agent}
            type="monotone"
            dataKey={agent}
            name={agent}
            stroke={agentColors[agent] ?? colors.inkLight}
            strokeWidth={2}
            dot={false}
          />
        ))}
      </LineChart>
    );
  }, [data, variant]);

  return (
    <div className="chart-card not-prose">
      <p className="mb-1 text-sm font-medium text-ink">{TITLES[variant]}</p>
      <p className="mb-4 text-sm text-ink-muted">{DESCRIPTIONS[variant]}</p>
      {error && (
        <p className="text-sm text-accent">Failed to load results: {error}</p>
      )}
      {!data && !error && (
        <p className="text-sm text-ink-light py-8 text-center">Loading experiment data…</p>
      )}
      {chartContent && (
        <div className="h-[280px] w-full">
          <ResponsiveContainer width="100%" height={280}>
            {chartContent}
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
