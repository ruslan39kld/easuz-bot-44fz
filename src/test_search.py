# src/test_search.py
import sys
import os

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hybrid_search import HybridSearch

# Поиск
hs = HybridSearch(index_dir="../data/easuz_index")
results = hs.search("ГРБС", top_k=3)

print("Результаты поиска:")
for i, r in enumerate(results, 1):
    print(f"{i}. {r['question'][:60]} | score={r['score']:.3f}")