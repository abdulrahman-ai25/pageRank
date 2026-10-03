# تنفيذ خوارزمية PageRank (Python)

تنفيذ بايثون لخوارزمية **PageRank** يطابق النموذج الرياضي ودراسة الحالة
الواردين في البحث *"خوارزمية PageRank ودورها في ترتيب الصفحات وتحليل
الشبكات"* (الفصول 3، 4، و6).

## بنية المشروع

```
pagerank_project/
├── pagerank/                  # الحزمة الأساسية
│   ├── __init__.py
│   ├── core.py                 # الخوارزمية: بناء M, Power Iteration, Personalized PageRank
│   └── visualize.py             # رسم بيان الشبكة (لإنتاج أشكال البحث)
├── examples/
│   ├── case_study.py            # إعادة إنتاج القسم 4 (دراسة الحالة A,B,C,D)
│   └── personalized_demo.py     # مثال على القسم 6 (Personalized PageRank)
├── tests/
│   └── test_pagerank.py         # اختبارات وحدة (pytest)
├── main.py                       # نقطة التشغيل الرئيسية
└── requirements.txt
```

## ربط الكود بفصول البحث

| دالة | القسم المقابل في البحث |
|---|---|
| `build_transition_matrix()` | 2.1، 2.4 (العقد الميتة)، 3.1 (مصفوفة الانتقال M) |
| `validate_transition_matrix()` | 3.1 — التحقق من أن M عمودية عشوائية |
| `pagerank()` | 3.3 (المعادلة الأساسية)، 3.4 (مصفوفة Google)، 3.5 (Power Iteration) |
| `rank_pages()` | 4.4 — تفسير النتائج وترتيب الصفحات |
| `personalized_pagerank()` | 6.1 — Personalized PageRank |
| `visualize.draw_graph()` | الأشكال 1، 2، 3 (مواضع `[أدخل الشكل رقم X هنا]`) |

## التثبيت والتشغيل

```bash
pip install -r requirements.txt

# تشغيل دراسة الحالة كاملة (يطابق جدول القسم 4.3)
python main.py

# تشغيل دراسة الحالة فقط
python -m examples.case_study

# تشغيل مثال Personalized PageRank فقط
python -m examples.personalized_demo

# تشغيل الاختبارات
pytest -v
```

## مثال استخدام مباشر

```python
import numpy as np
from pagerank import build_transition_matrix, pagerank, rank_pages

# تعريف الشبكة كما في القسم 4.1
adjacency = {
    "A": ["B", "C", "D"],
    "B": ["C", "D"],
    "C": ["A"],
    "D": ["C"],
}

# بناء مصفوفة الانتقال M (القسم 4.2)
M, pages = build_transition_matrix(adjacency, pages=["A", "B", "C", "D"])

# تشغيل Power Iteration مع d = 0.85 (القسم 4.3)
result = pagerank(M, d=0.85)

# الترتيب النهائي (القسم 4.4)
for page, score in rank_pages(result["v"], pages):
    print(f"{page}: {score:.4f}")

# الناتج المتوقع:
#   C: 0.3475
#   A: 0.3329
#   D: 0.1878
#   B: 0.1318
```

## ملاحظات حول التحقق من النتائج

- نتائج `examples/case_study.py` مُحقَّقة عبر مكتبة `networkx`
  (`networkx.pagerank`) في `tests/test_pagerank.py` كمرجع مستقل
  (`test_pagerank_matches_networkx`)، وتطابق القيم المُصحَّحة في
  القسم 4.3 من البحث (C ≈ 0.348، A ≈ 0.333، D ≈ 0.188، B ≈ 0.132).
- تمت معالجة العقد الميتة (Dangling Nodes، القسم 2.4) تلقائياً في
  `build_transition_matrix()`، حتى لو لم تُذكر الصفحة كمفتاح في
  قاموس الروابط.
- `validate_transition_matrix()` تتحقق من أن كل عمود في M يجمع إلى 1
  (الخاصية الأساسية لمصفوفة عمودية عشوائية، القسم 3.1) قبل تشغيل
  Power Iteration — مفيدة لاكتشاف أخطاء بناء الشبكة مبكراً.

## التوليد التلقائي للأشكال

يولّد `examples/case_study.py` تلقائياً صورة `output/figure_3_network.png`
تمثّل بيان الشبكة المدروسة مع تلوين كل عقدة بحسب قيمة PageRank الخاصة
بها، ويمكن إدراجها مباشرة في موضع `[أدخل الشكل رقم 3 هنا]` في ملف Word.
