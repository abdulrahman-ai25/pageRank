# -*- coding: utf-8 -*-
"""
personalized_demo.py — مثال على Personalized PageRank (القسم 6)
===================================================================

يستخدم شبكة القسم 4 نفسها، ويقارن:

1. PageRank الكلاسيكي (متجه قفز موحّد e = 1/N · 1).
2. Personalized PageRank بمتجه تفضيل يُركّز بالكامل على الصفحة A
   (e = [1, 0, 0, 0]) — كما لو أن مستخدماً يهتم بمواضيع الصفحة A فقط.

الهدف هو إظهار أن تغيير متجه التفضيل وحده يكفي لتغيير الترتيب
النهائي بالكامل، تماشياً مع ما ورد في القسم 6.3.

التشغيل:
    python -m examples.personalized_demo
"""

from __future__ import annotations

from pagerank import build_transition_matrix, pagerank, personalized_pagerank, rank_pages

ADJACENCY = {
    "A": ["B", "C", "D"],
    "B": ["C", "D"],
    "C": ["A"],
    "D": ["C"],
}
PAGES = ["A", "B", "C", "D"]
D_FACTOR = 0.85


def show_ranking(title: str, v, pages) -> None:
    print(title)
    for page, score in rank_pages(v, pages):
        print(f"    {page} : {score:.4f}")
    print()


def main() -> None:
    M, pages = build_transition_matrix(ADJACENCY, PAGES)

    # 1) Classic PageRank
    classic = pagerank(M, d=D_FACTOR)
    show_ranking("Classic PageRank  (e = 1/N for all pages):", classic["v"], pages)

    # 2) Personalized PageRank — full preference for page A (index 0)
    personalized = personalized_pagerank(M, preference={0: 1.0}, d=D_FACTOR)
    show_ranking(
        "Personalized PageRank  (full preference for page A, e = [1,0,0,0]):",
        personalized["v"], pages,
    )

    print("Difference (Personalized - Classic):")
    for page in pages:
        i = pages.index(page)
        diff = personalized["v"][i] - classic["v"][i]
        print(f"    {page} : {diff:+.4f}")


if __name__ == "__main__":
    main()
