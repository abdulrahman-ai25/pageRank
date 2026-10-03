# -*- coding: utf-8 -*-
"""
case_study.py — إعادة إنتاج دراسة الحالة في القسم 4 من البحث
================================================================

يعيد هذا الملف بناء شبكة الصفحات A, B, C, D من القسم 4.1، يحسب
مصفوفة الانتقال (القسم 4.2)، ثم يُشغّل Power Iteration مع d=0.85
(القسم 4.3) ويطبع جدولاً مطابقاً لجدول البحث، وينتهي بترتيب الصفحات
وتفسير مبسّط (القسم 4.4).

التشغيل:
    python -m examples.case_study
"""

from __future__ import annotations

import os

import numpy as np

from pagerank import (
    build_transition_matrix,
    pagerank,
    validate_transition_matrix,
    rank_pages,
)
from pagerank.visualize import draw_graph


# ────────────────────────────────────────────────────────────────
#  القسم 4.1 — تعريف الشبكة
# ────────────────────────────────────────────────────────────────
ADJACENCY = {
    "A": ["B", "C", "D"],   # A → B, C, D   (3 روابط خارجية)
    "B": ["C", "D"],        # B → C, D      (رابطان خارجيان)
    "C": ["A"],             # C → A          (رابط خارجي واحد)
    "D": ["C"],             # D → C          (رابط خارجي واحد)
}
PAGES = ["A", "B", "C", "D"]
D_FACTOR = 0.85


def print_matrix(M: np.ndarray, pages: list[str]) -> None:
    """Print transition matrix M as a readable table (Section 4.2)."""
    header = "          " + "   ".join(f"{p:>6}" for p in pages)
    print(header)
    for i, row_page in enumerate(pages):
        row = "   ".join(f"{M[i, j]:6.4f}" for j in range(len(pages)))
        print(f"  to {row_page} :  {row}")


def print_iteration_table(history: dict, pages: list[str],
                           rows: list[int | str]) -> None:
    """Print the iteration table (Section 4.3) for selected rows only."""
    header = f"{'Iteration':<12}" + "".join(f"{p:>10}" for p in pages)
    print(header)
    print("-" * len(header))
    for r in rows:
        if r not in history:
            continue
        v = history[r]
        label = f"k = {r}" if isinstance(r, int) else "converged"
        print(f"{label:<12}" + "".join(f"{val:>10.4f}" for val in v))


def main() -> None:
    print("=" * 60)
    print("Network Definition")
    print("=" * 60)
    for page, targets in ADJACENCY.items():
        print(f"  {page} -> {', '.join(targets)}  ({len(targets)} outgoing links)")

    # ── Section 4.2: Build transition matrix M ──────────────────
    print("\n" + "=" * 60)
    print("Transition Matrix M  (columns = source page)")
    print("=" * 60)
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    print_matrix(M, pages)

    validate_transition_matrix(M)
    print("\n✓ Check: every column of M sums to 1  (valid column-stochastic matrix)")

    # ── Section 4.3: Power Iteration with d = 0.85 ──────────────
    print("\n" + "=" * 60)
    print(f"Power Iteration  (d = {D_FACTOR})")
    print("=" * 60)
    result = pagerank(M, d=D_FACTOR, snapshot_at=(1, 2, 3, 5, 10))
    print_iteration_table(
        result["history"], pages,
        rows=[0, 1, 2, 3, 5, 10, "converged"],
    )
    print(f"\nIterations until convergence (eps = 1e-10): {result['iterations']}")

    # ── Section 4.4: Interpret results ──────────────────────────
    print("\n" + "=" * 60)
    print("Final Ranking")
    print("=" * 60)
    ranking = rank_pages(result["v"], pages)
    order_str = " > ".join(f"{p} ({score:.4f})" for p, score in ranking)
    print(f"  {order_str}")

    # ── Generate Figure 3 ────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "figure_3_network.png")

    pr_dict = dict(zip(pages, result["v"]))
    draw_graph(
        ADJACENCY, fig_path,
        pagerank_values=pr_dict,
        title="Figure 3 — Directed graph: pages A, B, C, D  (Section 4.1)",
    )
    print(f"\n✓ Figure 3 saved to: {os.path.abspath(fig_path)}")


if __name__ == "__main__":
    main()
