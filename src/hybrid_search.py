import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import re


class HybridSearch:
    """
    Гибридный поиск: FAISS (векторный) + BM25 (ключевые слова)
    + Фильтрация по категориям (44fz, 223fz, ARIP)
    
    Улучшения:
    - Динамический вес: короткие запросы (1-2 слова) → больше BM25
    - Фильтрация по категории
    - Дедупликация по source_file
    - Фильтрация по минимальному score
    """

    def __init__(self,
                 index_dir: str = "data/easuz_index",
                 model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> None:
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "faiss_index.bin"
        self.meta_path = self.index_dir / "index_metadata.pkl"

        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError(
                f"Не найдены файлы индекса в '{self.index_dir}'. Ожидаются: faiss_index.bin и index_metadata.pkl"
            )

        logging.info(f"[HybridSearch] Загрузка FAISS индекса из: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        logging.info(f"[HybridSearch] Загрузка метаданных из: {self.meta_path}")
        with open(self.meta_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self.ntotal = len(self.metadata)

        logging.info(f"[HybridSearch] Загрузка модели эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name)

        logging.info("[HybridSearch] Построение BM25 корпуса по метаданным индекса...")
        self.corpus_texts: List[str] = []
        for item in self.metadata:
            if item.get('type') == 'qa':
                text = f"{item.get('question', '')}\n{item.get('answer', '')}"
            else:
                text = item.get('text', '')
            self.corpus_texts.append(text)

        self.corpus_tokens = [self._tokenize_ru(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        # ⭐ НОВОЕ: Статистика по категориям
        categories_count = {}
        for m in self.metadata:
            cat = m.get('category', 'unknown')
            categories_count[cat] = categories_count.get(cat, 0) + 1
        
        logging.info(f"[HybridSearch] BM25 готов: документов = {len(self.corpus_tokens)}")
        logging.info(f"[HybridSearch] Категории: {categories_count}")

    def _tokenize_ru(self, text: str) -> List[str]:
        """Простая токенизация по словам (кириллица/латиница/цифры)"""
        return [w for w in re.findall(r"[\w]+", (text or '').lower()) if len(w) > 2]

    def _encode_query(self, query: str) -> np.ndarray:
        """Векторизация поискового запроса"""
        q = self.model.encode([query], normalize_embeddings=True)
        return q.astype('float32')

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Нормализация скоров в диапазон [0, 1]"""
        if scores.size == 0:
            return scores
        s_min = float(scores.min())
        s_max = float(scores.max())
        if s_max - s_min < 1e-8:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)

    def search(self, 
               query: str, 
               top_k: int = 5, 
               vector_weight: float = 0.7,
               category: str = None) -> List[Dict[str, Any]]:
        """
        Гибридный поиск с фильтрацией по категориям.
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            vector_weight: Вес векторного поиска (игнорируется, рассчитывается динамически)
            category: Фильтр по категории ("44fz", "223fz", "ARIP" или None для всех)
        
        Returns:
            Список результатов [{question, answer, source_file, category, score}]
        """
        if not query:
            return []

        # ⭐ ЛОГИРОВАНИЕ категории
        if category:
            logging.info(f"[HybridSearch] Поиск в категории: {category}")
        else:
            logging.info(f"[HybridSearch] Поиск по ВСЕМ категориям")

        # Динамический вес в зависимости от длины запроса
        query_tokens = self._tokenize_ru(query)
        if len(query_tokens) <= 2:
            vector_weight = 0.4
            logging.info(f"[HybridSearch] Короткий запрос ({len(query_tokens)} слов) → vector_weight=0.4")
        else:
            vector_weight = 0.7
            logging.info(f"[HybridSearch] Длинный запрос ({len(query_tokens)} слов) → vector_weight=0.7")

        # 1) Векторный поиск через FAISS
        q_vec = self._encode_query(query)
        faiss_k = min(max(top_k * 5, top_k), self.index.ntotal)
        sims, idxs = self.index.search(q_vec, faiss_k)
        vec_scores = sims[0]
        vec_idxs = idxs[0]

        vec_scores_norm = self._normalize(vec_scores)

        # 2) Ключевой поиск через BM25
        bm25_scores_all = np.array(self.bm25.get_scores(query_tokens), dtype=np.float32)
        bm25_top_k = min(faiss_k, self.ntotal)
        bm25_top_idxs = np.argsort(-bm25_scores_all)[:bm25_top_k]

        # 3) Объединяем кандидатов
        candidate_set = set(vec_idxs.tolist()) | set(bm25_top_idxs.tolist())
        
        # Нормализуем BM25 только по кандидатам
        candidate_list = list(candidate_set)
        candidate_bm25_scores = bm25_scores_all[candidate_list]
        candidate_bm25_norm = self._normalize(candidate_bm25_scores)
        bm25_score_map = dict(zip(candidate_list, candidate_bm25_norm))

        # 4) Вычисляем комбинированные скоры
        results: List[Dict[str, Any]] = []
        filtered_by_category = 0
        
        for i in candidate_set:
            m = self.metadata[i]
            
            # ⭐ ФИЛЬТРАЦИЯ ПО КАТЕГОРИИ
            doc_category = m.get('category', 'unknown')
            if category and doc_category != category:
                filtered_by_category += 1
                continue  # Пропускаем документы из других категорий
            
            # Векторный скор
            v_score = 0.0
            try:
                pos = int(np.where(vec_idxs == i)[0][0])
                v_score = float(vec_scores_norm[pos])
            except Exception:
                v_score = 0.0
            
            # BM25 скор из карты
            b_score = float(bm25_score_map.get(i, 0.0))

            combined = vector_weight * v_score + (1.0 - vector_weight) * b_score

            # Формируем результат
            if m.get('type') == 'qa':
                question = m.get('question', '')
                answer = m.get('answer', '')
                source = m.get('source', '')
            else:
                full_text = m.get('text', '')
                question = full_text[:200]
                answer = full_text
                source = m.get('source', '')

            results.append({
                'question': question,
                'answer': answer,
                'source_file': source,
                'category': doc_category,  # ⭐ СОХРАНЯЕМ КАТЕГОРИЮ
                'score': combined
            })

        # ⭐ ЛОГИРОВАНИЕ фильтрации
        if category:
            logging.info(f"[HybridSearch] Отфильтровано по категории: {filtered_by_category} документов")

        # 5) Сортируем
        results.sort(key=lambda x: x['score'], reverse=True)

        # 6) Фильтрация по порогу
        MIN_SCORE_THRESHOLD = 0.3
        results = [r for r in results if r['score'] >= MIN_SCORE_THRESHOLD]

        # 7) Дедупликация по source_file
        seen_sources = set()
        unique_results = []
        for res in results:
            src = res['source_file']
            if src not in seen_sources:
                seen_sources.add(src)
                unique_results.append(res)
                if len(unique_results) >= top_k:
                    break

        top = unique_results[:top_k]

        logging.info(
            f"[HybridSearch] query='{query[:80]}' | category={category or 'ALL'} | "
            f"candidates={len(candidate_set)} filtered={len(results)} -> unique={len(top)}"
        )
        if top:
            logging.info(
                f"[HybridSearch] TOP1: {top[0]['question'][:80]} | "
                f"category={top[0]['category']} | score={top[0]['score']:.3f}"
            )

        return top