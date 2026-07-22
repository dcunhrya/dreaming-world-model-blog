import type { Rng } from './rng';
import { choice } from './rng';

export type Pos = { r: number; c: number };
export type Action = 0 | 1 | 2 | 3; // up, right, down, left

export const NUM_ACTIONS = 4;

const DELTAS: Record<Action, Pos> = {
  0: { r: -1, c: 0 },
  1: { r: 0, c: 1 },
  2: { r: 1, c: 0 },
  3: { r: 0, c: -1 },
};

/** Default maze: vertical barrier at c=3 with gaps at (2,3) and (5,3). */
export function defaultWalls(size = 8): Pos[] {
  const walls: Pos[] = [];
  for (let r = 0; r < size; r++) {
    if (r !== 2 && r !== 5) {
      walls.push({ r, c: 3 });
    }
  }
  return walls;
}

export interface GridWorldOptions {
  size?: number;
  walls?: Pos[];
  start?: Pos;
  goal?: Pos;
  stepPenalty?: number;
  wallPenalty?: number;
  goalReward?: number;
  slipProb?: number;
  seed?: number;
}

export class GridWorld {
  readonly size: number;
  readonly walls: Set<string>;
  readonly start: Pos;
  readonly goal: Pos;
  readonly stepPenalty: number;
  readonly wallPenalty: number;
  readonly goalReward: number;
  readonly slipProb: number;
  private pos: Pos;
  private rng: Rng | null = null;

  constructor(opts: GridWorldOptions = {}) {
    this.size = opts.size ?? 8;
    const wallList = opts.walls ?? defaultWalls(this.size);
    this.walls = new Set(wallList.map((p) => this.key(p)));
    this.start = opts.start ?? { r: this.size - 1, c: 0 };
    this.goal = opts.goal ?? { r: 0, c: this.size - 1 };
    this.stepPenalty = opts.stepPenalty ?? -0.01;
    this.wallPenalty = opts.wallPenalty ?? -0.2;
    this.goalReward = opts.goalReward ?? 1.0;
    this.slipProb = opts.slipProb ?? 0.0;
    this.pos = { ...this.start };
  }

  setRng(rng: Rng): void {
    this.rng = rng;
  }

  key(p: Pos): string {
    return `${p.r},${p.c}`;
  }

  inBounds(p: Pos): boolean {
    return p.r >= 0 && p.r < this.size && p.c >= 0 && p.c < this.size;
  }

  isWall(p: Pos): boolean {
    return this.walls.has(this.key(p));
  }

  validStates(): Pos[] {
    const states: Pos[] = [];
    for (let r = 0; r < this.size; r++) {
      for (let c = 0; c < this.size; c++) {
        const p = { r, c };
        if (!this.isWall(p)) states.push(p);
      }
    }
    return states;
  }

  encode(p: Pos): number {
    return p.r * this.size + p.c;
  }

  decode(id: number): Pos {
    return { r: Math.floor(id / this.size), c: id % this.size };
  }

  nStates(): number {
    return this.size * this.size;
  }

  reset(): Pos {
    this.pos = { ...this.start };
    return { ...this.pos };
  }

  getPosition(): Pos {
    return { ...this.pos };
  }

  private resolveAction(a: Action): Action {
    if (this.slipProb > 0 && this.rng && this.rng() < this.slipProb) {
      const others = ([0, 1, 2, 3] as Action[]).filter((x) => x !== a);
      return choice(this.rng, others);
    }
    return a;
  }

  step(action: Action): {
    obs: Pos;
    reward: number;
    terminated: boolean;
    truncated: boolean;
  } {
    const a = this.resolveAction(action);
    const delta = DELTAS[a];
    const next = { r: this.pos.r + delta.r, c: this.pos.c + delta.c };

    if (!this.inBounds(next) || this.isWall(next)) {
      return {
        obs: { ...this.pos },
        reward: this.wallPenalty,
        terminated: false,
        truncated: false,
      };
    }

    this.pos = next;
    const atGoal = this.pos.r === this.goal.r && this.pos.c === this.goal.c;
    const reward = atGoal ? this.goalReward : this.stepPenalty;

    return {
      obs: { ...this.pos },
      reward,
      terminated: atGoal,
      truncated: false,
    };
  }
}
