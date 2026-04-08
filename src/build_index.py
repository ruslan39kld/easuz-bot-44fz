import json
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "easuz_index"

# Категории
CATEGORIES = {
    "44fz": DATA_DIR / "44fz",
    "223fz": DATA_DIR / "223fz", 
    "ARIP": DATA_DIR / "ARIP"
}

INDEX_DIR.mkdir(exist_ok=True)


def load_category_data(category_name: str, category_path: Path):
    """Загрузка данных одной категории"""
    print(f"\n📂 Загрузка категории: {category_name}")
    
    qa_file = category_path / "structured_qa.json"
    chunks_file = category_path / "chunks_data.json"
    
    metadata = []
    texts = []
    
    # Q&A
    if qa_file.exists():
        with open(qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f).get('qa_pairs', [])
        
        print(f"   ✅ Q&A: {len(qa_data)} пар")
        for item in qa_data:
            metadata.append({
                'type': 'qa',
                'category': category_name,  # ⭐ КАТЕГОРИЯ
                'question': item['question'],
                'answer': item['answer'],
                'source': item.get('source', 'unknown')
            })
            texts.append(f"{item['question']}\n{item['answer']}")
    
    # Chunks
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f).get('chunks', [])
        
        print(f"   ✅ Chunks: {len(chunks)} блоков")
        for item in chunks:
            metadata.append({
                'type': 'chunk',
                'category': category_name,  # ⭐ КАТЕГОРИЯ
                'text': item['text'],
                'source': item.get('source', 'unknown')
            })
            texts.append(item['text'])
    
    return texts, metadata


def build_index():
    """Создание индекса со всеми категориями"""
    print("=" * 60)
    print("🚀 СОЗДАНИЕ ИНДЕКСА С КАТЕГОРИЯМИ")
    print("=" * 60)
    
    all_texts = []
    all_metadata = []
    
    # Загрузка всех категорий
    for cat_name, cat_path in CATEGORIES.items():
        if not cat_path.exists():
            print(f"⚠️ Папка {cat_name} не найдена, пропускаем")
            continue
        
        texts, metadata = load_category_data(cat_name, cat_path)
        all_texts.extend(texts)
        all_metadata.extend(metadata)
    
    if not all_texts:
        print("❌ Нет данных для индексации!")
        return
    
    print(f"\n📊 Всего документов: {len(all_texts)}")
    print(f"   📝 По категориям:")
    for cat in ["44fz", "223fz", "ARIP"]:
        count = sum(1 for m in all_metadata if m['category'] == cat)
        print(f"      • {cat}: {count}")
    
    # Эмбеддинги
    print("\n🤖 Загрузка модели...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("🧠 Создание эмбеддингов...")
    embeddings = model.encode(
        all_texts, 
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )
    embeddings = embeddings.astype('float32')
    
    # FAISS индекс
    print("🔍 Создание FAISS индекса...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Сохранение
    print("💾 Сохранение...")
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.bin"))
    with open(INDEX_DIR / "index_metadata.pkl", 'wb') as f:
        pickle.dump(all_metadata, f)
    
    print("\n" + "=" * 60)
    print("✅ ИНДЕКС СОЗДАН")
    print("=" * 60)
    print(f"📍 Файлы: {INDEX_DIR}")
    print(f"📊 Векторов: {index.ntotal}")
    

if __name__ == "__main__":
    build_index()