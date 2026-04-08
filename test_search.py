import sys
import os

# Добавляем путь к src/ в sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_search import HybridSearch

# Инициализация поиска
hs = HybridSearch(index_dir="data/easuz_index")

# Поиск по запросу
results = hs.search("ГРБС", top_k=3)

print("\n✅ Результаты поиска:")
for i, r in enumerate(results, 1):
    print(f"  {i}. {r['question'][:60]} | score={r['score']:.3f}")