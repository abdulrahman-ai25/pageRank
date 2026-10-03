# -*- coding: utf-8 -*-
"""
visualize.py — رسم بيان الشبكة
=================================

يولّد رسماً بيانياً للشبكة الموجَّهة باستخدام networkx و matplotlib،
ويمكن استخدامه لإنتاج الأشكال المطلوبة في البحث:

    [أدخل الشكل رقم 1 هنا]  — تمثيل شبكة ويب بسيطة كبيان موجَّه (القسم 2.1)
    [أدخل الشكل رقم 3 هنا]  — بيان الشبكة المدروسة A,B,C,D (القسم 4.1)

يمكن أيضاً تلوين العقد حسب قيمة PageRank لإظهار النتائج بصرياً
(مفيد عند عرض القسم 4.4 أمام الدكتور).
"""

from __future__ import annotations

import networkx as nx
import matplotlib

matplotlib.use("Agg")  # رسم بدون واجهة تفاعلية (يناسب التنفيذ من سطر الأوامر)
import matplotlib.pyplot as plt


def draw_graph(adjacency: dict[str, list[str]],
                output_path: str,
                pagerank_values: dict[str, float] | None = None,
                title: str | None = None,
                seed: int = 42) -> None:
    """
    يرسم بيان الشبكة الموجَّه ويحفظه كملف صورة.

    Parameters
    ----------
    adjacency : dict[str, list[str]]
        قاموس {الصفحة: [الصفحات التي تشير إليها]}، بنفس صيغة
        build_transition_matrix().
    output_path : str
        مسار حفظ الصورة (مثال: "figure_3_network.png").
    pagerank_values : dict[str, float], اختياري
        إذا تم تمريره، تُلوَّن العقد وتُغيَّر أحجامها بحسب قيمة
        PageRank لكل صفحة (الأكبر = أهم = أكبر حجماً وأغمق لوناً).
    title : str, اختياري
        عنوان يُكتب أعلى الشكل.
    seed : int, افتراضي 42
        بذرة عشوائية لتخطيط العقد (لتكرار الشكل نفسه عند كل تشغيل).
    """
    G = nx.DiGraph()
    for source, targets in adjacency.items():
        G.add_node(source)
        for target in targets:
            G.add_edge(source, target)

    pos = nx.spring_layout(G, seed=seed, k=1.2)

    fig, ax = plt.subplots(figsize=(6, 5))

    if pagerank_values:
        # Keep node size fixed so arrows remain readable; PageRank is still
        # conveyed through color and numeric labels.
        fixed_node_size = 3600
        colors = [pagerank_values.get(n, 0) for n in G.nodes()]
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=fixed_node_size, node_color=colors,
            cmap=plt.cm.Blues, vmin=0, vmax=max(pagerank_values.values()) * 1.2,
            edgecolors="#1F3864", linewidths=1.5,
        )
        fig.colorbar(nodes, ax=ax, label="PageRank", shrink=0.8)
    else:
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=2200, node_color="#BDD7EE",
            edgecolors="#1F3864", linewidths=1.5,
        )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_weight="bold")

    # Keep arrowheads outside the node boundary so direction stays visible
    # even when PageRank nodes are drawn with large sizes.
    nx.draw_networkx_edges(
        G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=30,
        connectionstyle="arc3,rad=0.12", edge_color="#555555",
        width=1.5, min_source_margin=25, min_target_margin=25,
    )

    if pagerank_values:
        labels = {n: f"{pagerank_values.get(n, 0):.3f}" for n in G.nodes()}
        label_pos = {n: (x, y - 0.14) for n, (x, y) in pos.items()}
        for n, (x, y) in label_pos.items():
            ax.text(x, y, labels[n], ha="center", fontsize=9, color="#1F3864")

    if title:
        ax.set_title(title, fontsize=13)

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
