import { useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { CategoricalChartState } from 'recharts/types/chart/types';
import { chartTooltipStyle, colors } from '../../lib/theme';

export type MultiAgentCurvePoint = { x: number; y: number };

export type MultiAgentCurve = {
  agent: string;
  points: MultiAgentCurvePoint[];
};

function nearestPoint(points: MultiAgentCurvePoint[], x: number): MultiAgentCurvePoint | null {
  if (points.length === 0) return null;
  return points.reduce((best, p) => (Math.abs(p.x - x) < Math.abs(best.x - x) ? p : best));
}

function AllAgentsTooltip({
  x,
  curves,
  xLabel,
  agentStroke,
}: {
  x: number;
  curves: MultiAgentCurve[];
  xLabel: string;
  agentStroke: (agent: string) => string;
}) {
  return (
    <div style={chartTooltipStyle} className="rounded-lg px-3 py-2 text-xs shadow-sm">
      <p className="mb-1.5 font-medium text-ink">
        {xLabel}: {Math.round(x).toLocaleString()}
      </p>
      <ul className="space-y-1">
        {curves.map((curve) => {
          const pt = nearestPoint(curve.points, x);
          if (pt == null) return null;
          return (
            <li key={curve.agent} className="flex items-center gap-2 text-ink-muted">
              <span
                className="inline-block h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: agentStroke(curve.agent) }}
              />
              <span>
                {curve.agent}: <strong className="text-ink">{pt.y.toFixed(2)}</strong>
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export default function MultiAgentLearningChart({
  curves,
  xLabel,
  agentStroke,
}: {
  curves: MultiAgentCurve[];
  xLabel: string;
  agentStroke: (agent: string) => string;
}) {
  const [hoverX, setHoverX] = useState<number | null>(null);

  const handleMouseMove = (state: CategoricalChartState) => {
    const xAxis = state?.xAxisMap ? Object.values(state.xAxisMap)[0] : undefined;
    const scale = xAxis?.scale as ((v: number) => number) & { invert?: (p: number) => number };
    if (state?.activeCoordinate && scale?.invert) {
      const x = scale.invert(state.activeCoordinate.x);
      if (Number.isFinite(x)) setHoverX(x);
      return;
    }
    if (state?.activeLabel != null) {
      setHoverX(Number(state.activeLabel));
    }
  };

  return (
    <LineChart
      margin={{ top: 8, right: 8, bottom: 24, left: 8 }}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setHoverX(null)}
    >
      <CartesianGrid strokeDasharray="3 3" stroke={colors.chartGrid} />
      <XAxis
        type="number"
        dataKey="x"
        domain={[0, 'auto']}
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
      <Tooltip
        cursor={{ stroke: colors.inkLight, strokeDasharray: '4 4' }}
        content={() =>
          hoverX != null ? (
            <AllAgentsTooltip
              x={hoverX}
              curves={curves}
              xLabel={xLabel}
              agentStroke={agentStroke}
            />
          ) : null
        }
        isAnimationActive={false}
      />
      {curves.map((curve) => (
        <Line
          key={curve.agent}
          data={curve.points}
          type="monotone"
          dataKey="y"
          name={curve.agent}
          stroke={agentStroke(curve.agent)}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      ))}
    </LineChart>
  );
}

export function toMultiAgentCurves(
  curves: { agent: string; points: { x: number; mean?: number; meanReturn?: number }[] }[]
): MultiAgentCurve[] {
  return curves.map((curve) => ({
    agent: curve.agent,
    points: curve.points.map((p) => ({
      x: p.x,
      y: p.mean ?? p.meanReturn ?? 0,
    })),
  }));
}
