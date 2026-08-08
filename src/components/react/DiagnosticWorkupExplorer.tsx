import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { INTERACTIVE_DIAGNOSTIC_PROFILE } from '../../dreaming/diagnostic/config';
import { DiagnosticWorkupEnv } from '../../dreaming/diagnostic/env';
import { replayLearnedCase, trainDiagnosticAsync } from '../../dreaming/diagnostic/replay';
import {
  DISEASE_LABELS,
  Disease,
  RISK_LABELS,
  RiskProfile,
  TEST_LABELS,
  Test,
  TestResult,
} from '../../dreaming/diagnostic/types';
import { colors, chartTooltipStyle } from '../../lib/theme';

const K_OPTIONS = [0, 5, 20] as const;
const BIAS_OPTIONS = [0, 0.25, 0.5, 0.75] as const;
const PE_OPTIONS = [0.5, 1, 2, 4] as const;

function ResultBadge({ result }: { result: TestResult }) {
  const label =
    result === TestResult.Unobserved
      ? 'Pending'
      : result === TestResult.Negative
        ? 'Negative'
        : 'Positive';
  const cls =
    result === TestResult.Positive
      ? 'bg-red-100 text-red-900'
      : result === TestResult.Negative
        ? 'bg-emerald-100 text-emerald-900'
        : 'bg-stone-100 text-stone-600';
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${cls}`}>{label}</span>;
}

function DifferentialChart({ posterior }: { posterior: Record<Disease, number> }) {
  const data = [Disease.ACS, Disease.PE, Disease.Pneumonia, Disease.Benign].map((d) => ({
    disease: DISEASE_LABELS[d],
    p: posterior[d],
  }));
  return (
    <div className="h-[180px] w-full">
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
          <XAxis dataKey="disease" tick={{ fill: colors.inkMuted, fontSize: 11 }} />
          <YAxis domain={[0, 1]} tick={{ fill: colors.inkMuted, fontSize: 11 }} />
          <Tooltip contentStyle={chartTooltipStyle} formatter={(v: number) => v.toFixed(2)} />
          <Bar dataKey="p" name="Simulator posterior" fill={colors.cardinal} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function CasePanel({
  title,
  replay,
  stepIndex,
}: {
  title: string;
  replay: ReturnType<typeof replayLearnedCase> | null;
  stepIndex: number;
}) {
  if (!replay) {
    return (
      <div className="rounded-lg border border-border bg-cream-card p-4 text-sm text-ink-light">
        {title}: training…
      </div>
    );
  }
  const step = replay.steps[Math.min(stepIndex, replay.steps.length - 1)];
  const state = step?.state;
  return (
    <div className="rounded-lg border border-border bg-cream-card p-4">
      <p className="mb-2 text-sm font-semibold text-ink">{title}</p>
      <p className="mb-3 text-xs text-ink-muted">
        Hidden truth: <strong>{replay.trueDiseaseLabel}</strong> · Return:{' '}
        {replay.totalReturn.toFixed(2)} · {replay.correct ? 'Correct' : 'Incorrect'}
      </p>
      {step && (
        <>
          <p className="mb-2 text-sm text-ink">
            Step {stepIndex + 1}: <strong>{step.actionLabel}</strong> (r = {step.reward.toFixed(2)})
          </p>
          <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
            {[Test.ECG, Test.Troponin, Test.DDimer, Test.CXR].map((t) => (
              <div key={t} className="flex items-center justify-between rounded bg-cream-muted/50 px-2 py-1">
                <span>{TEST_LABELS[t]}</span>
                <ResultBadge result={state?.results[t] ?? TestResult.Unobserved} />
              </div>
            ))}
          </div>
          <p className="mb-1 text-xs font-medium text-ink-muted">
            Simulator-derived differential (explanation only)
          </p>
          <DifferentialChart posterior={step.posterior} />
        </>
      )}
    </div>
  );
}

export default function DiagnosticWorkupExplorer() {
  const [kIndex, setKIndex] = useState(2);
  const [biasIndex, setBiasIndex] = useState(0);
  const [peIndex, setPeIndex] = useState(1);
  const [risk, setRisk] = useState<RiskProfile>(RiskProfile.Medium);
  const [seed, setSeed] = useState(42);
  const [stepIndex, setStepIndex] = useState(0);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');
  const [qLearningReplay, setQLearningReplay] = useState<ReturnType<
    typeof replayLearnedCase
  > | null>(null);
  const [dynaReplay, setDynaReplay] = useState<ReturnType<typeof replayLearnedCase> | null>(
    null
  );

  const dreamK = K_OPTIONS[kIndex] ?? 20;
  const modelBias = BIAS_OPTIONS[biasIndex] ?? 0;
  const peMultiplier = PE_OPTIONS[peIndex] ?? 1;

  const env = useMemo(
    () =>
      new DiagnosticWorkupEnv({
        pePrevalenceMultiplier: peMultiplier,
        modelBiasLevel: modelBias,
      }),
    [peMultiplier, modelBias]
  );

  const runDemo = useCallback(async (runSeed: number = seed) => {
    setRunning(true);
    setProgress('Training Q-learning…');
    setQLearningReplay(null);
    setDynaReplay(null);
    setStepIndex(0);

    const trainCfg = {
      episodes: INTERACTIVE_DIAGNOSTIC_PROFILE.episodes,
      dreamUpdatesPerStep: 0,
      modelBiasLevel: modelBias,
      pePrevalenceMultiplier: peMultiplier,
      riskProfile: risk,
    };

    const q0 = await trainDiagnosticAsync(trainCfg, runSeed, (ep, total) =>
      setProgress(`Training Q-learning… ${ep}/${total}`)
    );
    setQLearningReplay(replayLearnedCase(env, q0, runSeed + 1000, risk));

    setProgress(`Training Dyna-Q (k=${dreamK})…`);
    const qDyna = await trainDiagnosticAsync(
      { ...trainCfg, dreamUpdatesPerStep: dreamK },
      runSeed,
      (ep, total) => setProgress(`Training Dyna-Q… ${ep}/${total}`)
    );
    setDynaReplay(replayLearnedCase(env, qDyna, runSeed + 1000, risk));
    setProgress('');
    setRunning(false);
  }, [dreamK, env, modelBias, peMultiplier, risk, seed]);

  useEffect(() => {
    runDemo();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const maxSteps = Math.max(qLearningReplay?.steps.length ?? 0, dynaReplay?.steps.length ?? 0);

  return (
    <div className="chart-card not-prose">
      <p className="mb-4 text-sm text-ink-muted">
        Synthetic ED chest-pain workup. The agent chooses tests or a final diagnosis. The{' '}
        <strong>ground-truth simulator</strong> uses hand-specified disease priors and test
        sensitivities — no EHR data. The <strong>learned world model</strong> is built from
        rollouts, then used for Dyna-Q dreaming.
      </p>

      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <label className="block text-sm">
          <span className="font-medium text-ink">Dream budget k = {dreamK}</span>
          <input
            type="range"
            min={0}
            max={K_OPTIONS.length - 1}
            value={kIndex}
            disabled={running}
            onChange={(e) => setKIndex(Number(e.target.value))}
            className="mt-1 w-full accent-accent"
          />
        </label>
        <label className="block text-sm">
          <span className="font-medium text-ink">Model bias = {modelBias.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={BIAS_OPTIONS.length - 1}
            value={biasIndex}
            disabled={running}
            onChange={(e) => setBiasIndex(Number(e.target.value))}
            className="mt-1 w-full accent-amber-600"
          />
        </label>
        <label className="block text-sm">
          <span className="font-medium text-ink">PE prevalence ×{peMultiplier}</span>
          <input
            type="range"
            min={0}
            max={PE_OPTIONS.length - 1}
            value={peIndex}
            disabled={running}
            onChange={(e) => setPeIndex(Number(e.target.value))}
            className="mt-1 w-full accent-accent"
          />
        </label>
        <label className="block text-sm">
          <span className="font-medium text-ink">Risk profile</span>
          <select
            value={risk}
            disabled={running}
            onChange={(e) => setRisk(Number(e.target.value) as RiskProfile)}
            className="mt-1 w-full rounded border border-border bg-cream-card px-2 py-1"
          >
            {[RiskProfile.Low, RiskProfile.Medium, RiskProfile.High].map((r) => (
              <option key={r} value={r}>
                {RISK_LABELS[r]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={running}
          onClick={() => runDemo(seed)}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {running ? 'Training…' : 'Run comparison'}
        </button>
        <button
          type="button"
          disabled={running}
          onClick={() => {
            const s = Math.floor(Math.random() * 10000);
            setSeed(s);
            runDemo(s);
          }}
          className="rounded-lg border border-border px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          New seed
        </button>
        <span className="text-xs text-ink-light">Seed: {seed}</span>
        {progress && <span className="text-xs text-ink-muted">{progress}</span>}
      </div>

      {maxSteps > 0 && (
        <label className="mb-6 block text-sm">
          <span className="font-medium text-ink">
            Case step {stepIndex + 1} / {maxSteps}
          </span>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxSteps - 1)}
            value={stepIndex}
            onChange={(e) => setStepIndex(Number(e.target.value))}
            className="mt-1 w-full accent-accent"
          />
        </label>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <CasePanel title="Q-learning (k = 0)" replay={qLearningReplay} stepIndex={stepIndex} />
        <CasePanel title={`Dyna-Q (k = ${dreamK})`} replay={dynaReplay} stepIndex={stepIndex} />
      </div>
    </div>
  );
}
