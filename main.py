# -*- coding: utf-8 -*-
"""
main.py — نقطة التشغيل الرئيسية
==================================

يشغّل دراسة الحالة الكاملة (القسم 4 من البحث) ثم مثال
Personalized PageRank (القسم 6).

التشغيل:
    python main.py
"""

from examples import case_study, personalized_demo


def main() -> None:
    case_study.main()

    print("\n\n")
    print("#" * 60)
    print("  Bonus: Personalized PageRank (Section 6)")
    print("#" * 60 + "\n")

    personalized_demo.main()


if __name__ == "__main__":
    main()
