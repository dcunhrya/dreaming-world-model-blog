import type { Action } from './env';
import type { Rng } from './rng';

export type QTable = Float64Array[];

export const DEFAULT_ALPHA = 0.1;
export const DEFAULT_GAMMA = 0.95;
export const DEFAULT_EPSILON = 0.1;

export function createQ(nStates: number, nActions: number): QTable {
  const Q: QTable = [];
  for (let s = 0; s < nStates; s++) {
    Q.push(new Float64Array(nActions));
  }
  return Q;
}

export function maxQ(Q: QTable, s: number): number {
  let best = -Infinity;
  for (let a = 0; a < Q[s]!.length; a++) {
    if (Q[s]![a]! > best) best = Q[s]![a]!;
  }
  return best === -Infinity ? 0 : best;
}

export function argmaxQ(Q: QTable, s: number): Action {
  let bestA: Action = 0;
  let best = -Infinity;
  for (let a = 0; a < Q[s]!.length; a++) {
    if (Q[s]![a]! > best) {
      best = Q[s]![a]!;
      bestA = a as Action;
    }
  }
  return bestA;
}

export function qUpdate(
  Q: QTable,
  s: number,
  a: number,
  r: number,
  sNext: number,
  alpha: number,
  gamma: number,
  done: boolean
): void {
  const bestNext = done ? 0 : maxQ(Q, sNext);
  const tdTarget = r + gamma * bestNext;
  Q[s]![a]! += alpha * (tdTarget - Q[s]![a]!);
}

export function epsilonGreedy(
  Q: QTable,
  s: number,
  epsilon: number,
  rng: Rng
): Action {
  if (rng() < epsilon) {
    return Math.floor(rng() * 4) as Action;
  }
  return argmaxQ(Q, s);
}
