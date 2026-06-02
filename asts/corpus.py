"""Realistic Python and TypeScript code prefixes for the microbenchmark.

We embed samples directly (no network) at four prefix-length tiers, so the
benchmark is hermetic and reproducible. Sizes target ~10/~100/~1000/~5000
tokens (approximated as ~4 chars/token for code).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    name: str
    language: str  # "python" | "typescript"
    code: str
    approx_tokens: int  # rough estimate for sorting/labeling


# ---------------------------------------------------------------------------
# Python samples
# ---------------------------------------------------------------------------

PY_TINY = '''def add(a: int, b: int) -> int:
    return a + b
'''

PY_SMALL = '''import json
from pathlib import Path

def load_config(path: str) -> dict:
    """Load a JSON config file from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_config(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
'''

PY_MEDIUM = '''from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence


@dataclass
class Vector:
    """A simple n-dimensional vector with common operations."""

    components: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[float]:
        return iter(self.components)

    def __add__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("dimension mismatch")
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("dimension mismatch")
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other: "Vector") -> float:
        if len(self) != len(other):
            raise ValueError("dimension mismatch")
        return sum(a * b for a, b in zip(self.components, other.components))

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.components))

    def normalized(self) -> "Vector":
        n = self.norm()
        if n == 0.0:
            raise ValueError("cannot normalize zero vector")
        return Vector([x / n for x in self.components])

    def scale(self, k: float) -> "Vector":
        return Vector([x * k for x in self.components])

    @classmethod
    def zeros(cls, dim: int) -> "Vector":
        return cls([0.0] * dim)


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity between two non-zero vectors."""
    na, nb = a.norm(), b.norm()
    if na == 0.0 or nb == 0.0:
        raise ValueError("zero vector")
    return a.dot(b) / (na * nb)


def mean_vector(vs: Sequence[Vector]) -> Vector:
    if not vs:
        raise ValueError("empty sequence")
    dim = len(vs[0])
    acc = [0.0] * dim
    for v in vs:
        if len(v) != dim:
            raise ValueError("dimension mismatch")
        for i, x in enumerate(v):
            acc[i] += x
    n = len(vs)
    return Vector([x / n for x in acc])
'''

PY_LARGE = (
    PY_MEDIUM
    + '''

@dataclass
class Matrix:
    """A simple dense row-major matrix with common operations."""

    rows: list[list[float]]

    def __post_init__(self) -> None:
        if not self.rows:
            self.n_rows = 0
            self.n_cols = 0
            return
        self.n_rows = len(self.rows)
        self.n_cols = len(self.rows[0])
        for r in self.rows:
            if len(r) != self.n_cols:
                raise ValueError("ragged matrix")

    @classmethod
    def zeros(cls, n_rows: int, n_cols: int) -> "Matrix":
        return cls([[0.0] * n_cols for _ in range(n_rows)])

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        m = cls.zeros(n, n)
        for i in range(n):
            m.rows[i][i] = 1.0
        return m

    def transpose(self) -> "Matrix":
        out = Matrix.zeros(self.n_cols, self.n_rows)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                out.rows[j][i] = self.rows[i][j]
        return out

    def matmul(self, other: "Matrix") -> "Matrix":
        if self.n_cols != other.n_rows:
            raise ValueError("shape mismatch for matmul")
        out = Matrix.zeros(self.n_rows, other.n_cols)
        for i in range(self.n_rows):
            for k in range(self.n_cols):
                a = self.rows[i][k]
                if a == 0.0:
                    continue
                for j in range(other.n_cols):
                    out.rows[i][j] += a * other.rows[k][j]
        return out

    def matvec(self, v: Vector) -> Vector:
        if len(v) != self.n_cols:
            raise ValueError("shape mismatch for matvec")
        out = [0.0] * self.n_rows
        for i in range(self.n_rows):
            row = self.rows[i]
            s = 0.0
            for j in range(self.n_cols):
                s += row[j] * v.components[j]
            out[i] = s
        return Vector(out)

    def add(self, other: "Matrix") -> "Matrix":
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise ValueError("shape mismatch for add")
        out = Matrix.zeros(self.n_rows, self.n_cols)
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                out.rows[i][j] = self.rows[i][j] + other.rows[i][j]
        return out

    def scale(self, k: float) -> "Matrix":
        return Matrix([[x * k for x in row] for row in self.rows])

    def trace(self) -> float:
        if self.n_rows != self.n_cols:
            raise ValueError("trace requires square matrix")
        return sum(self.rows[i][i] for i in range(self.n_rows))

    def frobenius_norm(self) -> float:
        return math.sqrt(sum(x * x for row in self.rows for x in row))


def gram_schmidt(vectors: Sequence[Vector]) -> list[Vector]:
    """Classical Gram-Schmidt orthonormalization. Skips zero residuals."""
    out: list[Vector] = []
    for v in vectors:
        u = v
        for q in out:
            u = u - q.scale(q.dot(v))
        n = u.norm()
        if n > 1e-12:
            out.append(u.scale(1.0 / n))
    return out


def power_iteration(m: Matrix, n_iters: int = 50, tol: float = 1e-9) -> tuple[float, Vector]:
    """Estimate the dominant eigenpair of a square matrix via power iteration."""
    if m.n_rows != m.n_cols:
        raise ValueError("power iteration requires square matrix")
    n = m.n_rows
    v = Vector([1.0 / math.sqrt(n)] * n)
    eigval = 0.0
    for _ in range(n_iters):
        w = m.matvec(v)
        new_eig = v.dot(w)
        nw = w.norm()
        if nw < 1e-30:
            break
        new_v = w.scale(1.0 / nw)
        if abs(new_eig - eigval) < tol:
            eigval = new_eig
            v = new_v
            break
        eigval, v = new_eig, new_v
    return eigval, v
'''
)

# Build a ~5000-token large sample by replicating with renames.
def _replicate_python(base: str, copies: int) -> str:
    parts = [base]
    for i in range(2, copies + 1):
        # Mangle class names so the result is still valid Python (no redefinition issues
        # except for top-level dataclasses, which Python allows but linters dislike;
        # the parse benchmark only cares about syntactic shape, not execution).
        suffix = f"_v{i}"
        renamed = (
            base.replace("class Vector", f"class Vector{suffix}")
            .replace('"Vector"', f'"Vector{suffix}"')
            .replace("Vector(", f"Vector{suffix}(")
            .replace("class Matrix", f"class Matrix{suffix}")
            .replace('"Matrix"', f'"Matrix{suffix}"')
            .replace("Matrix(", f"Matrix{suffix}(")
            .replace("def cosine_similarity", f"def cosine_similarity{suffix}")
            .replace("def mean_vector", f"def mean_vector{suffix}")
            .replace("def gram_schmidt", f"def gram_schmidt{suffix}")
            .replace("def power_iteration", f"def power_iteration{suffix}")
        )
        parts.append(renamed)
    return "\n\n".join(parts)


PY_HUGE = _replicate_python(PY_LARGE, 3)


# ---------------------------------------------------------------------------
# TypeScript samples
# ---------------------------------------------------------------------------

TS_TINY = '''export function add(a: number, b: number): number {
  return a + b;
}
'''

TS_SMALL = '''import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

export interface Config {
  [key: string]: unknown;
}

export function loadConfig(path: string): Config {
  const text = readFileSync(path, "utf-8");
  return JSON.parse(text) as Config;
}

export function saveConfig(path: string, data: Config): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(data, null, 2), "utf-8");
}
'''

TS_MEDIUM = '''export type Vec = ReadonlyArray<number>;

export class Vector {
  readonly components: number[];

  constructor(components: number[]) {
    this.components = components.slice();
  }

  get dim(): number {
    return this.components.length;
  }

  static zeros(dim: number): Vector {
    return new Vector(new Array<number>(dim).fill(0));
  }

  add(other: Vector): Vector {
    if (this.dim !== other.dim) {
      throw new Error("dimension mismatch");
    }
    const out = new Array<number>(this.dim);
    for (let i = 0; i < this.dim; i++) {
      out[i] = this.components[i] + other.components[i];
    }
    return new Vector(out);
  }

  sub(other: Vector): Vector {
    if (this.dim !== other.dim) {
      throw new Error("dimension mismatch");
    }
    const out = new Array<number>(this.dim);
    for (let i = 0; i < this.dim; i++) {
      out[i] = this.components[i] - other.components[i];
    }
    return new Vector(out);
  }

  dot(other: Vector): number {
    if (this.dim !== other.dim) {
      throw new Error("dimension mismatch");
    }
    let s = 0;
    for (let i = 0; i < this.dim; i++) {
      s += this.components[i] * other.components[i];
    }
    return s;
  }

  norm(): number {
    let s = 0;
    for (const x of this.components) s += x * x;
    return Math.sqrt(s);
  }

  scale(k: number): Vector {
    return new Vector(this.components.map((x) => x * k));
  }

  normalized(): Vector {
    const n = this.norm();
    if (n === 0) throw new Error("cannot normalize zero vector");
    return this.scale(1 / n);
  }
}

export function cosineSimilarity(a: Vector, b: Vector): number {
  const na = a.norm();
  const nb = b.norm();
  if (na === 0 || nb === 0) throw new Error("zero vector");
  return a.dot(b) / (na * nb);
}

export function meanVector(vs: ReadonlyArray<Vector>): Vector {
  if (vs.length === 0) throw new Error("empty sequence");
  const dim = vs[0].dim;
  const acc = new Array<number>(dim).fill(0);
  for (const v of vs) {
    if (v.dim !== dim) throw new Error("dimension mismatch");
    for (let i = 0; i < dim; i++) acc[i] += v.components[i];
  }
  return new Vector(acc.map((x) => x / vs.length));
}
'''

TS_LARGE = (
    TS_MEDIUM
    + '''

export class Matrix {
  readonly rows: number[][];
  readonly nRows: number;
  readonly nCols: number;

  constructor(rows: number[][]) {
    this.rows = rows.map((r) => r.slice());
    this.nRows = rows.length;
    this.nCols = rows.length === 0 ? 0 : rows[0].length;
    for (const r of rows) {
      if (r.length !== this.nCols) throw new Error("ragged matrix");
    }
  }

  static zeros(nRows: number, nCols: number): Matrix {
    const rows: number[][] = [];
    for (let i = 0; i < nRows; i++) {
      rows.push(new Array<number>(nCols).fill(0));
    }
    return new Matrix(rows);
  }

  static identity(n: number): Matrix {
    const m = Matrix.zeros(n, n);
    for (let i = 0; i < n; i++) m.rows[i][i] = 1;
    return m;
  }

  transpose(): Matrix {
    const out = Matrix.zeros(this.nCols, this.nRows);
    for (let i = 0; i < this.nRows; i++) {
      for (let j = 0; j < this.nCols; j++) {
        out.rows[j][i] = this.rows[i][j];
      }
    }
    return out;
  }

  matmul(other: Matrix): Matrix {
    if (this.nCols !== other.nRows) throw new Error("shape mismatch");
    const out = Matrix.zeros(this.nRows, other.nCols);
    for (let i = 0; i < this.nRows; i++) {
      for (let k = 0; k < this.nCols; k++) {
        const a = this.rows[i][k];
        if (a === 0) continue;
        for (let j = 0; j < other.nCols; j++) {
          out.rows[i][j] += a * other.rows[k][j];
        }
      }
    }
    return out;
  }

  matvec(v: Vector): Vector {
    if (v.dim !== this.nCols) throw new Error("shape mismatch");
    const out = new Array<number>(this.nRows).fill(0);
    for (let i = 0; i < this.nRows; i++) {
      let s = 0;
      for (let j = 0; j < this.nCols; j++) s += this.rows[i][j] * v.components[j];
      out[i] = s;
    }
    return new Vector(out);
  }

  add(other: Matrix): Matrix {
    if (this.nRows !== other.nRows || this.nCols !== other.nCols) {
      throw new Error("shape mismatch");
    }
    const out = Matrix.zeros(this.nRows, this.nCols);
    for (let i = 0; i < this.nRows; i++) {
      for (let j = 0; j < this.nCols; j++) {
        out.rows[i][j] = this.rows[i][j] + other.rows[i][j];
      }
    }
    return out;
  }

  scale(k: number): Matrix {
    return new Matrix(this.rows.map((r) => r.map((x) => x * k)));
  }

  trace(): number {
    if (this.nRows !== this.nCols) throw new Error("trace requires square matrix");
    let s = 0;
    for (let i = 0; i < this.nRows; i++) s += this.rows[i][i];
    return s;
  }

  frobeniusNorm(): number {
    let s = 0;
    for (const r of this.rows) for (const x of r) s += x * x;
    return Math.sqrt(s);
  }
}

export function gramSchmidt(vectors: ReadonlyArray<Vector>): Vector[] {
  const out: Vector[] = [];
  for (const v of vectors) {
    let u = v;
    for (const q of out) {
      u = u.sub(q.scale(q.dot(v)));
    }
    const n = u.norm();
    if (n > 1e-12) out.push(u.scale(1 / n));
  }
  return out;
}

export function powerIteration(
  m: Matrix,
  nIters: number = 50,
  tol: number = 1e-9,
): { eigenvalue: number; eigenvector: Vector } {
  if (m.nRows !== m.nCols) throw new Error("power iteration requires square matrix");
  const n = m.nRows;
  let v = new Vector(new Array<number>(n).fill(1 / Math.sqrt(n)));
  let eigval = 0;
  for (let it = 0; it < nIters; it++) {
    const w = m.matvec(v);
    const newEig = v.dot(w);
    const nw = w.norm();
    if (nw < 1e-30) break;
    const newV = w.scale(1 / nw);
    if (Math.abs(newEig - eigval) < tol) {
      eigval = newEig;
      v = newV;
      break;
    }
    eigval = newEig;
    v = newV;
  }
  return { eigenvalue: eigval, eigenvector: v };
}
'''
)


def _replicate_typescript(base: str, copies: int) -> str:
    parts = [base]
    for i in range(2, copies + 1):
        suffix = f"V{i}"
        renamed = (
            base.replace("export class Vector", f"export class Vector{suffix}")
            .replace("new Vector(", f"new Vector{suffix}(")
            .replace(": Vector", f": Vector{suffix}")
            .replace("export class Matrix", f"export class Matrix{suffix}")
            .replace("new Matrix", f"new Matrix{suffix}")
            .replace(": Matrix", f": Matrix{suffix}")
            .replace("export function cosineSimilarity", f"export function cosineSimilarity{suffix}")
            .replace("export function meanVector", f"export function meanVector{suffix}")
            .replace("export function gramSchmidt", f"export function gramSchmidt{suffix}")
            .replace("export function powerIteration", f"export function powerIteration{suffix}")
        )
        parts.append(renamed)
    return "\n\n".join(parts)


TS_HUGE = _replicate_typescript(TS_LARGE, 3)


# ---------------------------------------------------------------------------
# Public catalog
# ---------------------------------------------------------------------------

def _approx_tokens(s: str) -> int:
    # Crude but standard: ~4 chars/token for code-tokenized text.
    return max(1, len(s) // 4)


SAMPLES: list[Sample] = [
    Sample("py-tiny", "python", PY_TINY, _approx_tokens(PY_TINY)),
    Sample("py-small", "python", PY_SMALL, _approx_tokens(PY_SMALL)),
    Sample("py-medium", "python", PY_MEDIUM, _approx_tokens(PY_MEDIUM)),
    Sample("py-large", "python", PY_LARGE, _approx_tokens(PY_LARGE)),
    Sample("py-huge", "python", PY_HUGE, _approx_tokens(PY_HUGE)),
    Sample("ts-tiny", "typescript", TS_TINY, _approx_tokens(TS_TINY)),
    Sample("ts-small", "typescript", TS_SMALL, _approx_tokens(TS_SMALL)),
    Sample("ts-medium", "typescript", TS_MEDIUM, _approx_tokens(TS_MEDIUM)),
    Sample("ts-large", "typescript", TS_LARGE, _approx_tokens(TS_LARGE)),
    Sample("ts-huge", "typescript", TS_HUGE, _approx_tokens(TS_HUGE)),
]


def get_samples(language: str | None = None) -> list[Sample]:
    if language is None:
        return list(SAMPLES)
    return [s for s in SAMPLES if s.language == language]


def get_sample(name: str) -> Sample:
    for s in SAMPLES:
        if s.name == name:
            return s
    raise KeyError(name)
