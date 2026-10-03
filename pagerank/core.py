# -*- coding: utf-8 -*-
"""
core.py — التنفيذ الأساسي لخوارزمية PageRank
================================================

يطبّق هذا الملف الصياغة الرياضية الكاملة من الفصل الثالث في البحث:

    القسم 3.1  build_transition_matrix()  → بناء مصفوفة الانتقال M
    القسم 3.3  pagerank()                  → المعادلة الأساسية v = dMv + (1-d)/N·1
    القسم 3.5  pagerank()                  → خوارزمية Power Iteration
    القسم 6.1  personalized_pagerank()     → v = dMv + (1-d)·e

تم تجنّب الحلقات المتداخلة (nested loops) في خطوة التكرار، واستُخدمت
عمليات مصفوفات numpy مباشرة تماشياً مع الصياغة v = G·v.
"""

from __future__ import annotations

import numpy as np


# ────────────────────────────────────────────────────────────────
#  القسم 2.1 / 3.1 — بناء مصفوفة الانتقال M من بنية الروابط
# ────────────────────────────────────────────────────────────────
def build_transition_matrix(adjacency: dict[str, list[str]],
                             pages: list[str] | None = None
                             ) -> tuple[np.ndarray, list[str]]:
    """
    يبني مصفوفة الانتقال M (column-stochastic) من قائمة ارتباط.

    التعريف الرياضي (القسم 3.1):

        M[i][j] = 1 / L(j)   إذا كانت الصفحة j تشير إلى الصفحة i
        M[i][j] = 0          خلاف ذلك

    حيث L(j) هو عدد الروابط الصادرة من الصفحة j.

    معالجة العقد الميتة (Dangling Nodes — القسم 2.4):
    إذا لم تحتوِ الصفحة j على أي رابط صادر (L(j) = 0)، يُفترض أنها
    ترتبط بجميع الصفحات بالتساوي، أي M[i][j] = 1/N لكل i. هذا يحافظ
    على خاصية أن مجموع كل عمود يساوي 1.

    Parameters
    ----------
    adjacency : dict[str, list[str]]
        قاموس {الصفحة: [قائمة الصفحات التي تشير إليها]}.
    pages : list[str], اختياري
        ترتيب الصفحات في المصفوفة. إذا لم يُحدَّد، يُستخرج تلقائياً
        من مفاتيح adjacency بترتيبها.

    Returns
    -------
    M : np.ndarray
        مصفوفة الانتقال بأبعاد N×N.
    pages : list[str]
        ترتيب الصفحات المستخدَم (يفيد لتفسير عناصر متجه v).

    Examples
    --------
    >>> adjacency = {"A": ["B", "C", "D"], "B": ["C", "D"],
    ...              "C": ["A"], "D": ["C"]}
    >>> M, pages = build_transition_matrix(adjacency, ["A", "B", "C", "D"])
    >>> np.allclose(M.sum(axis=0), 1.0)   # كل عمود يجمع إلى 1
    True
    """
    if pages is None:
        pages = list(adjacency.keys())

    n = len(pages)
    index = {page: i for i, page in enumerate(pages)}
    M = np.zeros((n, n))

    # ملاحظة: التكرار على `pages` (لا على adjacency.items()) ضروري لضمان
    # معالجة العقد الميتة (القسم 2.4) حتى لو لم تُذكر الصفحة كمفتاح في
    # القاموس أصلاً (وهي الحالة الشائعة عند استخراج الروابط من بيانات حقيقية).
    for source in pages:
        j = index[source]
        targets = adjacency.get(source, [])
        out_degree = len(targets)

        if out_degree == 0:
            # عقدة ميتة: توزيع موحّد على كل الصفحات (القسم 2.4)
            M[:, j] = 1.0 / n
        else:
            weight = 1.0 / out_degree
            for target in targets:
                if target not in index:
                    raise ValueError(
                        f"الصفحة '{source}' تشير إلى '{target}' وهي غير "
                        f"موجودة في قائمة الصفحات pages={pages}"
                    )
                i = index[target]
                M[i, j] += weight

    return M, pages


# ────────────────────────────────────────────────────────────────
#  القسم 3.3 / 3.5 — المعادلة الأساسية + Power Iteration
# ────────────────────────────────────────────────────────────────
def pagerank(M: np.ndarray,
             d: float = 0.85,
             eps: float = 1e-10,
             max_iter: int = 200,
             snapshot_at: tuple[int, ...] = ()
             ) -> dict:
    """
    يحسب متجه PageRank عبر خوارزمية Power Iteration.

    المعادلة المُطبَّقة في كل تكرار (القسم 3.3):

        v_(k+1) = d · M · v_(k) + (1-d)/N · 1

    وهي مكافئة لـ v_(k+1) = G · v_(k) حيث G مصفوفة Google (القسم 3.4):

        G = d·M + (1-d)/N · E

    شرط التوقف (القسم 3.5):

        ‖ v_(k+1) − v_(k) ‖₁ < ε

    Parameters
    ----------
    M : np.ndarray
        مصفوفة الانتقال (column-stochastic) N×N.
    d : float, افتراضي 0.85
        عامل التخميد (القسم 2.6).
    eps : float, افتراضي 1e-10
        عتبة التقارب ε (القسم 3.5).
    max_iter : int, افتراضي 200
        أقصى عدد تكرارات مسموح به.
    snapshot_at : tuple[int, ...], اختياري
        أرقام تكرارات يُراد تسجيل v عندها في history (لإعادة بناء
        جدول مثل جدول القسم 4.3).

    Returns
    -------
    dict
        - "v"          : متجه PageRank النهائي (بعد التقارب)
        - "iterations" : عدد التكرارات الفعلي حتى التقارب
        - "history"    : dict {رقم التكرار: نسخة من v عند تلك اللحظة}
        - "converged"  : True إذا تحقق شرط التقارب قبل max_iter

    Notes
    -----
    تتحقق العملية من أن مجموع v يبقى 1 (متجه احتمالي) عبر كل التكرارات،
    لأن G مصفوفة عمودية عشوائية بالكامل (القسم 3.4).
    """
    n = M.shape[0]
    e_uniform = np.full(n, 1.0 / n)  # متجه القفز العشوائي الموحّد (1/N · 1)

    v = e_uniform.copy()  # v^(0) = (1/N, ..., 1/N)ᵗ — القسم 3.2
    history = {0: v.copy()}

    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        v_next = d * (M @ v) + (1 - d) * e_uniform

        if k in snapshot_at:
            history[k] = v_next.copy()

        if np.linalg.norm(v_next - v, ord=1) < eps:
            v = v_next
            converged = True
            break

        v = v_next

    history["converged"] = v.copy()

    return {
        "v": v,
        "iterations": k,
        "history": history,
        "converged": converged,
    }


# ────────────────────────────────────────────────────────────────
#  أدوات مساعدة: التحقق من المصفوفة وترجمة v إلى ترتيب صفحات
# ────────────────────────────────────────────────────────────────
def validate_transition_matrix(M: np.ndarray, atol: float = 1e-9) -> None:
    """
    يتحقق من أن M مصفوفة عمودية عشوائية صالحة (القسم 3.1):

    - مصفوفة مربعة N×N.
    - جميع العناصر غير سالبة.
    - مجموع كل عمود يساوي 1 تماماً (بعد معالجة العقد الميتة، القسم 2.4).

    يرفع ValueError مع رسالة توضيحية إذا فشل أي شرط. مفيد كخطوة فحص
    سريعة قبل تمرير M إلى pagerank(), خصوصاً عند بناء الشبكة يدوياً.

    Raises
    ------
    ValueError
        إذا كانت M غير مربعة، أو تحتوي على قيم سالبة، أو كان أحد
        الأعمدة لا يجمع إلى 1.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M يجب أن تكون مصفوفة مربعة، الأبعاد الحالية: {M.shape}")

    if np.any(M < -atol):
        raise ValueError("M تحتوي على عناصر سالبة، وهذا مخالف لتعريف مصفوفة الانتقال")

    col_sums = M.sum(axis=0)
    bad_cols = np.where(np.abs(col_sums - 1.0) > atol)[0]
    if bad_cols.size > 0:
        details = ", ".join(f"العمود {j}: المجموع={col_sums[j]:.6f}" for j in bad_cols)
        raise ValueError(
            "كل عمود في M يجب أن يجمع إلى 1 (القسم 3.1). "
            f"أعمدة غير صحيحة → {details}"
        )


def rank_pages(v: np.ndarray, pages: list[str]) -> list[tuple[str, float]]:
    """
    يربط متجه PageRank v بأسماء الصفحات ويُرتّبها تنازلياً.

    تُعيد قائمة (اسم الصفحة، الدرجة) مرتبة من الأعلى إلى الأقل،
    وهو الشكل المُستخدَم في تفسير النتائج بالقسم 4.4
    (مثال: C > A > D > B).

    Parameters
    ----------
    v : np.ndarray
        متجه PageRank (الناتج من pagerank()["v"]).
    pages : list[str]
        أسماء الصفحات بنفس الترتيب المُستخدَم في بناء M.

    Returns
    -------
    list[tuple[str, float]]
        قائمة (الصفحة, الدرجة) مرتبة تنازلياً حسب الدرجة.

    Examples
    --------
    >>> rank_pages(np.array([0.33, 0.13, 0.35, 0.19]), ["A", "B", "C", "D"])
    [('C', 0.35), ('A', 0.33), ('D', 0.19), ('B', 0.13)]
    """
    if len(v) != len(pages):
        raise ValueError(
            f"طول v ({len(v)}) لا يطابق عدد الصفحات ({len(pages)})"
        )
    return sorted(zip(pages, v), key=lambda item: item[1], reverse=True)


# ────────────────────────────────────────────────────────────────
#  القسم 6.1 — Personalized PageRank
# ────────────────────────────────────────────────────────────────
def personalized_pagerank(M: np.ndarray,
                           preference: dict[int, float] | np.ndarray,
                           d: float = 0.85,
                           eps: float = 1e-10,
                           max_iter: int = 200
                           ) -> dict:
    """
    يحسب Personalized PageRank باستخدام متجه تفضيل e (القسم 6.1):

        v_(k+1) = d · M · v_(k) + (1-d) · e

    خلافاً للنموذج الكلاسيكي حيث e[i] = 1/N للجميع، يسمح هذا الامتداد
    بإعطاء قيم أعلى للصفحات التي تهمّ مستخدماً معيناً، مما يُنتج ترتيباً
    شخصياً (انظر تطبيقاته في أنظمة التوصية، القسم 6.2).

    Parameters
    ----------
    M : np.ndarray
        مصفوفة الانتقال N×N.
    preference : dict[int, float] أو np.ndarray
        متجه التفضيل e. يمكن تمريره كمصفوفة numpy مباشرة بطول N،
        أو كقاموس {فهرس الصفحة: وزن} يُطبَّع تلقائياً ليصبح مجموعه 1
        (الصفحات غير المذكورة تأخذ وزناً صفرياً).
    d, eps, max_iter
        كما في pagerank().

    Returns
    -------
    dict
        نفس مخرجات pagerank() (v, iterations, history, converged).

    Examples
    --------
    >>> # تفضيل شخصي يركّز بالكامل على الصفحة A (الفهرس 0)
    >>> result = personalized_pagerank(M, preference={0: 1.0})
    """
    n = M.shape[0]

    if isinstance(preference, dict):
        e = np.zeros(n)
        for idx, weight in preference.items():
            e[idx] = weight
    else:
        e = np.asarray(preference, dtype=float)
        if e.shape != (n,):
            raise ValueError(
                f"طول متجه التفضيل ({e.shape}) لا يطابق عدد الصفحات N={n}"
            )

    if e.sum() == 0:
        raise ValueError("متجه التفضيل لا يمكن أن يكون كله أصفاراً")
    if np.any(e < 0):
        raise ValueError("عناصر متجه التفضيل يجب أن تكون غير سالبة")

    e = e / e.sum()  # تطبيع: Σe[i] = 1

    v = np.full(n, 1.0 / n)
    history = {0: v.copy()}

    converged = False
    k = 0
    for k in range(1, max_iter + 1):
        v_next = d * (M @ v) + (1 - d) * e

        if np.linalg.norm(v_next - v, ord=1) < eps:
            v = v_next
            converged = True
            break

        v = v_next

    history["converged"] = v.copy()

    return {
        "v": v,
        "iterations": k,
        "history": history,
        "converged": converged,
    }
