# -*- coding: utf-8 -*-
"""
test_pagerank.py — اختبارات وحدة لحزمة pagerank
====================================================

تغطي هذه الاختبارات:
- بناء مصفوفة الانتقال (الحالة العادية والعقد الميتة).
- خاصية كون M عمودية عشوائية (column-stochastic).
- صحة نتائج دراسة الحالة في القسم 4 (مقارنة بالقيم المحسوبة بـ numpy
  مباشرة، وبمكتبة networkx كمرجع مستقل).
- Personalized PageRank: التطبيع وحالة الخطأ عند e = 0.
- التحقق من صحة المدخلات (validate_transition_matrix).

التشغيل:
    pytest -v
"""

import numpy as np
import pytest

from pagerank import (
    build_transition_matrix,
    pagerank,
    personalized_pagerank,
    validate_transition_matrix,
    rank_pages,
)


ADJACENCY = {
    "A": ["B", "C", "D"],
    "B": ["C", "D"],
    "C": ["A"],
    "D": ["C"],
}
PAGES = ["A", "B", "C", "D"]


# ────────────────────────────────────────────────────────────────
#  build_transition_matrix
# ────────────────────────────────────────────────────────────────
def test_transition_matrix_values():
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    assert pages == PAGES

    expected = np.array([
        [0,   0,   1, 0],
        [1/3, 0,   0, 0],
        [1/3, 1/2, 0, 1],
        [1/3, 1/2, 0, 0],
    ])
    np.testing.assert_allclose(M, expected)


def test_transition_matrix_columns_sum_to_one():
    M, _ = build_transition_matrix(ADJACENCY, PAGES)
    np.testing.assert_allclose(M.sum(axis=0), np.ones(4))


def test_dangling_node_handled():
    """صفحة بلا روابط خارجية (D) يجب أن تُوزَّع بالتساوي على الجميع."""
    adjacency = {"A": ["B"], "B": ["A"], "C": ["A"], "D": []}
    pages = ["A", "B", "C", "D"]
    M, _ = build_transition_matrix(adjacency, pages)

    # عمود D (الفهرس 3) يجب أن يكون 1/4 لكل صف
    np.testing.assert_allclose(M[:, 3], np.full(4, 0.25))
    np.testing.assert_allclose(M.sum(axis=0), np.ones(4))


def test_dangling_node_not_listed_as_key():
    """عقدة ميتة غير مذكورة كمفتاح في adjacency يجب أن تُعالَج أيضاً."""
    adjacency = {"A": ["B"], "B": ["A"]}  # C غير مذكورة على الإطلاق
    pages = ["A", "B", "C"]
    M, _ = build_transition_matrix(adjacency, pages)

    np.testing.assert_allclose(M[:, 2], np.full(3, 1 / 3))
    np.testing.assert_allclose(M.sum(axis=0), np.ones(3))


def test_invalid_target_raises():
    adjacency = {"A": ["X"], "B": ["A"]}
    with pytest.raises(ValueError):
        build_transition_matrix(adjacency, ["A", "B"])


# ────────────────────────────────────────────────────────────────
#  validate_transition_matrix
# ────────────────────────────────────────────────────────────────
def test_validate_accepts_correct_matrix():
    M, _ = build_transition_matrix(ADJACENCY, PAGES)
    validate_transition_matrix(M)  # لا يجب أن يرفع أي استثناء


def test_validate_rejects_non_square():
    with pytest.raises(ValueError):
        validate_transition_matrix(np.ones((3, 4)))


def test_validate_rejects_bad_column_sum():
    M = np.array([[0.5, 0.0], [0.5, 0.5]])  # العمود 1 يجمع إلى 0.5
    with pytest.raises(ValueError):
        validate_transition_matrix(M)


def test_validate_rejects_negative_entries():
    M = np.array([[1.0, -0.5], [0.0, 1.5]])
    with pytest.raises(ValueError):
        validate_transition_matrix(M)


# ────────────────────────────────────────────────────────────────
#  pagerank — دراسة الحالة (القسم 4)
# ────────────────────────────────────────────────────────────────
def test_pagerank_converges_and_sums_to_one():
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    result = pagerank(M, d=0.85)

    assert result["converged"] is True
    np.testing.assert_allclose(result["v"].sum(), 1.0)


def test_pagerank_case_study_values():
    """مقارنة بالقيم المُصحَّحة في القسم 4.3 من البحث (دقة 3 أرقام عشرية)."""
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    result = pagerank(M, d=0.85)
    v = dict(zip(pages, result["v"]))

    expected = {"A": 0.3329, "B": 0.1318, "C": 0.3475, "D": 0.1878}
    for page, exp_val in expected.items():
        assert v[page] == pytest.approx(exp_val, abs=1e-3)


def test_pagerank_ranking_order():
    """الترتيب النهائي يجب أن يكون C > A > D > B (القسم 4.4)."""
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    result = pagerank(M, d=0.85)
    ranking = rank_pages(result["v"], pages)
    order = [page for page, _ in ranking]
    assert order == ["C", "A", "D", "B"]


def test_pagerank_first_iteration_manual():
    """تحقق من قيمة v(C) عند k=1 المذكورة في القسم 4.3 (≈ 0.4271)."""
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    result = pagerank(M, d=0.85, snapshot_at=(1,))
    v1 = dict(zip(pages, result["history"][1]))
    assert v1["C"] == pytest.approx(0.4271, abs=1e-3)


def test_pagerank_matches_networkx():
    """مقارنة مستقلة بمكتبة networkx (تُستخدَم كمرجع صناعي للتحقق)."""
    networkx = pytest.importorskip("networkx")

    G = networkx.DiGraph()
    for source, targets in ADJACENCY.items():
        for target in targets:
            G.add_edge(source, target)

    nx_pr = networkx.pagerank(G, alpha=0.85, tol=1e-12)

    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    result = pagerank(M, d=0.85, eps=1e-12)
    v = dict(zip(pages, result["v"]))

    for page in pages:
        assert v[page] == pytest.approx(nx_pr[page], abs=1e-4)


# ────────────────────────────────────────────────────────────────
#  personalized_pagerank — القسم 6
# ────────────────────────────────────────────────────────────────
def test_personalized_pagerank_sums_to_one():
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    result = personalized_pagerank(M, preference={0: 1.0}, d=0.85)
    np.testing.assert_allclose(result["v"].sum(), 1.0)


def test_personalized_pagerank_shifts_ranking():
    """تفضيل كامل لصفحة B يجب أن يرفع درجتها مقارنةً بالنموذج الكلاسيكي."""
    M, pages = build_transition_matrix(ADJACENCY, PAGES)

    classic = pagerank(M, d=0.85)
    personalized = personalized_pagerank(M, preference={1: 1.0}, d=0.85)  # B

    b_index = pages.index("B")
    assert personalized["v"][b_index] > classic["v"][b_index]


def test_personalized_pagerank_rejects_zero_preference():
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    with pytest.raises(ValueError):
        personalized_pagerank(M, preference=np.zeros(4), d=0.85)


def test_personalized_pagerank_rejects_wrong_length():
    M, pages = build_transition_matrix(ADJACENCY, PAGES)
    with pytest.raises(ValueError):
        personalized_pagerank(M, preference=np.array([1.0, 0.0]), d=0.85)


# ────────────────────────────────────────────────────────────────
#  rank_pages
# ────────────────────────────────────────────────────────────────
def test_rank_pages_order_and_length_check():
    v = np.array([0.33, 0.13, 0.35, 0.19])
    ranking = rank_pages(v, PAGES)
    assert [p for p, _ in ranking] == ["C", "A", "D", "B"]

    with pytest.raises(ValueError):
        rank_pages(v, ["A", "B"])  # طول غير مطابق
