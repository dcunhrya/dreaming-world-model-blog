import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { GridWorld } from '../../dreaming/env';
import {
  INTERACTIVE_PROFILE,
  rollingMean,
  runSingle,
  runTrainingAsync,
  snapshotMaxQGrid,
  type MaxQGrid,
} from '../../dreaming/runExperiment';
import type { EpisodeMetrics } from '../../dreaming/dyna';
import { colors, chartTooltipStyle } from '../../lib/theme';

const K_OPTIONS = [0, 1, 5, 20, 100] as const;
const NOISE_OPTIONS = [0, 0.05, 0.1, 0.2] as const;

function QHeatmap({
  grid,
  env,
  title,
  min,
  max,
}: {
  grid: MaxQGrid;
  env: GridWorld;
  title: string;
  min: number;
  max: number;
}) {
  const range = max - min || 1;

  return (
    <div className="flex flex-col items-center">
      <p className="mb-2 text-sm font-medium text-ink">{title}</p>
      <div
        className="inline-grid gap-0.5"
        style={{
          gridTemplateColumns: `repeat(${env.size}, minmax(0, 1.75rem))`,
        }}
      >
        {grid.grid.map((row, r) =>
          row.map((val, c) => {
            const p = { r, c };
            const isWall = env.isWall(p);
            const isGoal = p.r === env.goal.r && p.c === env.goal.c;
            const isStart = p.r === env.start.r && p.c === env.start.c;
            let bg = 'rgb(241 245 249)';
            if (!isWall && Number.isFinite(val)) {
              const t = (val - min) / range;
              const hue = 220 - t * 120;
              bg = `hsl(${hue} 70% ${45 + t * 25}%)`;
            }
            return (
              <div
                key={`${r}-${c}`}
                className="relative flex h-7 w-7 items-center justify-center rounded-sm text-[9px] font-medium text-white"
                style={{ backgroundColor: isWall ? 'rgb(100 116 139)' : bg }}
                title={isWall ? 'wall' : `max Q = ${val.toFixed(2)}`}
              >
                {isGoal && <span className="text-white drop-shadow">G</span>}
                {isStart && !isGoal && (
                  <span className="text-white drop-shadow">S</span>
                )}
              </div>
            );
          })
        )}
      </div>
      <p className="mt-2 text-xs text-ink-light">Color = max<sub>a</sub> Q(s, a)</p>
    </div>
  );
}

export default function DynaQExplorer() {
  const [kIndex, setKIndex] = useState(3); // k=20
  const [noiseIndex, setNoiseIndex] = useState(0);
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState<EpisodeMetrics[]>([]);
  const [baselineHistory, setBaselineHistory] = useState<EpisodeMetrics[]>([]);
  const [currentGrid, setCurrentGrid] = useState<MaxQGrid | null>(null);
  const [baselineGrid, setBaselineGrid] = useState<MaxQGrid | null>(null);
  const cancelRef = useRef(false);

  const dreamK = K_OPTIONS[kIndex] ?? 20;
  const modelNoise = NOISE_OPTIONS[noiseIndex] ?? 0;
  const env = useMemo(() => new GridWorld(), []);

  const runComparison = useCallback(async (runSeed: number, kVal: number, noiseVal: number) => {
    cancelRef.current = false;
    setRunning(true);
    setHistory([]);
    setBaselineHistory([]);
    setCurrentGrid(null);
    setBaselineGrid(null);

    const episodes = INTERACTIVE_PROFILE.episodes;

    // Run k=0 baseline synchronously (fast enough)
    const baseline = runSingle(
      { episodes, dreamUpdatesPerStep: 0, modelNoise: 0, env: new GridWorld() },
      runSeed
    );
    setBaselineHistory(baseline.history);
    setBaselineGrid(snapshotMaxQGrid(baseline.finalQ, env));

    // Run current settings with chunked async updates
    await runTrainingAsync(
      {
        episodes,
        dreamUpdatesPerStep: kVal,
        modelNoise: noiseVal,
        env: new GridWorld(),
      },
      runSeed,
      8,
      (hist, Q, gridEnv, done) => {
        setHistory([...hist]);
        if (done) {
          setCurrentGrid(snapshotMaxQGrid(Q, gridEnv));
          setRunning(false);
        }
      },
      () => cancelRef.current
    );
  }, [env]);

  useEffect(() => {
    runComparison(42, 20, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRun = () => {
    runComparison(seed, dreamK, modelNoise);
  };

  const handleReset = () => {
    cancelRef.current = true;
    setRunning(false);
    const newSeed = Math.floor(Math.random() * 10000);
    setSeed(newSeed);
    runComparison(newSeed, dreamK, modelNoise);
  };

  const chartData = useMemo(() => {
    const len = Math.max(history.length, baselineHistory.length);
    const data = [];
    const smoothWindow = 5;
    const smoothBaseline = rollingMean(
      baselineHistory.map((h) => h.episodeReturn),
      smoothWindow
    );
    const smoothCurrent = rollingMean(
      history.map((h) => h.episodeReturn),
      smoothWindow
    );
    for (let i = 0; i < len; i++) {
      data.push({
        episode: i + 1,
        baseline: smoothBaseline[i] ?? null,
        current: smoothCurrent[i] ?? null,
      });
    }
    return data;
  }, [history, baselineHistory]);

  return (
    <div className="chart-card not-prose">
      <p className="mb-4 text-sm text-ink-muted">
        Adjust the <strong>dreaming budget</strong> (imagined Q-updates per real step) and{' '}
        <strong>model noise</strong>, then run Dyna-Q on the 8×8 GridWorld. Compare learning
        curves and max-Q heatmaps: left is Q-learning (k=0), right is your settings.
      </p>

      <div className="mb-6 grid gap-6 sm:grid-cols-2">
        <label className="block">
          <span className="mb-1 block text-sm font-medium text-ink">
            Dream updates per real step: <strong>k = {dreamK}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={K_OPTIONS.length - 1}
            step={1}
            value={kIndex}
            disabled={running}
            onChange={(e) => setKIndex(Number(e.target.value))}
            className="w-full accent-accent"
          />
          <div className="mt-1 flex justify-between text-xs text-ink-light">
            {K_OPTIONS.map((k) => (
              <span key={k}>{k}</span>
            ))}
          </div>
        </label>

        <label className="block">
          <span className="mb-1 block text-sm font-medium text-ink">
            Model noise: <strong>{modelNoise.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min={0}
            max={NOISE_OPTIONS.length - 1}
            step={1}
            value={noiseIndex}
            disabled={running}
            onChange={(e) => setNoiseIndex(Number(e.target.value))}
            className="w-full accent-amber-600"
          />
          <div className="mt-1 flex justify-between text-xs text-ink-light">
            {NOISE_OPTIONS.map((n) => (
              <span key={n}>{n}</span>
            ))}
          </div>
        </label>
      </div>

      <div className="mb-6 flex flex-wrap gap-3">
        <button
          type="button"
          onClick={handleRun}
          disabled={running}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-dim disabled:opacity-50"
        >
          {running ? 'Training…' : 'Run'}
        </button>
        <button
          type="button"
          onClick={handleReset}
          disabled={running}
          className="rounded-lg border border-border bg-cream-card px-4 py-2 text-sm font-medium text-ink hover:bg-cream-muted/80 disabled:opacity-50"
        >
          Reset seed
        </button>
        <span className="self-center text-xs text-ink-light">Seed: {seed}</span>
      </div>

      <div className="mb-8 h-[220px] w-full">
        <p className="mb-2 text-sm font-medium text-ink">
          Smoothed episode return (baseline k=0 vs current)
        </p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
            <XAxis
              dataKey="episode"
              tick={{ fill: colors.inkMuted, fontSize: 11 }}
              label={{ value: 'Episode', position: 'insideBottom', offset: -4, fontSize: 11 }}
            />
            <YAxis tick={{ fill: colors.inkMuted, fontSize: 11 }} />
            <Tooltip contentStyle={chartTooltipStyle} />
            <Line
              type="monotone"
              dataKey="baseline"
              name="k=0"
              stroke={colors.inkLight}
              strokeWidth={2}
              dot={false}
            />
            <Line
              type="monotone"
              dataKey="current"
              name={`k=${dreamK}`}
              stroke={colors.cardinal}
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {baselineGrid && (
          <QHeatmap
            grid={baselineGrid}
            env={env}
            title="Q-learning (k = 0)"
            min={baselineGrid.min}
            max={baselineGrid.max}
          />
        )}
        {currentGrid ? (
          <QHeatmap
            grid={currentGrid}
            env={env}
            title={`Dyna-Q (k = ${dreamK}, noise = ${modelNoise})`}
            min={currentGrid.min}
            max={currentGrid.max}
          />
        ) : (
          <div className="flex items-center justify-center text-sm text-ink-light">
            {running ? 'Training…' : 'Run to see heatmap'}
          </div>
        )}
      </div>
    </div>
  );
}
