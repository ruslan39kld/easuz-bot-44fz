# check_index.py
import pickle
import os

INDEX_DIR = "data/easuz_index"
METADATA_FILE = os.path.join(INDEX_DIR, "index_metadata.pkl")

with open(METADATA_FILE, 'rb') as f:
    metadata = pickle.load(f)

print("✅ Метаданные загружены")
print(f"Количество элементов: {len(metadata)}")
print("Первые 3 элемента:")
for i, item in enumerate(metadata[:3], 1):
    print(f"  {i}. {item}")