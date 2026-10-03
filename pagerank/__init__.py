"""
pagerank — تنفيذ خوارزمية PageRank
=====================================

حزمة بايثون تطبّق النموذج الرياضي الموصوف في الفصل الثالث من البحث:

    v = d · M · v + (1-d)/N · 1        (المعادلة الأساسية، القسم 3.3)
    G = d · M + (1-d)/N · E            (مصفوفة Google، القسم 3.4)
    v = G · v   via Power Iteration    (القسم 3.5)
"""

from .core import (
    build_transition_matrix,
    pagerank,
    personalized_pagerank,
    validate_transition_matrix,
    rank_pages,
)

__all__ = [
    "build_transition_matrix",
    "pagerank",
    "personalized_pagerank",
    "validate_transition_matrix",
    "rank_pages",
]

__version__ = "1.0.0"
