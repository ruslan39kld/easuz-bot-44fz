# -*- coding: utf-8 -*-
"""
Обработка данных по категориям (44fz, 223fz, ARIP)
Создает structured_qa.json и chunks_data.json для каждой категории
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import PyPDF2
import docx

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Категории
CATEGORIES = ["44fz", "223fz", "ARIP"]


def process_excel_category(category: str):
    """Обработка Excel файлов одной категории"""
    print(f"\n📊 Обработка Excel для {category}...")
    
    excel_dir = DATA_DIR / category / "excel_qa"
    if not excel_dir.exists():
        print(f"   ⚠️ Папка {excel_dir} не найдена")
        return []
    
    excel_files = list(excel_dir.glob("*.xlsx"))
    print(f"   Найдено файлов: {len(excel_files)}")
    
    qa_pairs = []
    
    for file in tqdm(excel_files, desc=f"  {category}"):
        try:
            df = pd.read_excel(file, header=None)
            
            if df.empty or len(df.columns) < 3:
                continue
            
            # Поиск заголовков
            header_row = None
            question_col = None
            answer_col = None
            
            for idx, row in df.iterrows():
                row_str = ' '.join([str(x).lower() for x in row if pd.notna(x)])
                if 'вопрос' in row_str and 'ответ' in row_str:
                    header_row = idx
                    for col_idx, val in enumerate(row):
                        val_str = str(val).lower()
                        if 'вопрос' in val_str:
                            question_col = col_idx
                        if 'ответ' in val_str:
                            answer_col = col_idx
                    break
            
            # Стандарт
            if question_col is None or answer_col is None:
                question_col = 1
                answer_col = 2
                header_row = 0
            
            # Обработка строк
            start_row = header_row + 1 if header_row is not None else 1
            
            for idx in range(start_row, len(df)):
                try:
                    q = str(df.iloc[idx, question_col]).strip()
                    a = str(df.iloc[idx, answer_col]).strip()
                    
                    if (q and a and 
                        q not in ['nan', 'None', '', 'NaT'] and 
                        a not in ['nan', 'None', '', 'NaT'] and
                        not q.isdigit() and
                        len(q) > 5 and len(a) > 10):
                        
                        qa_pairs.append({
                            "id": f"{category}_qa_{len(qa_pairs)+1:04d}",
                            "question": q.lower(),
                            "answer": a,
                            "source": file.name,
                            "category": category  # ⭐ КАТЕГОРИЯ
                        })
                except (IndexError, KeyError):
                    continue
                        
        except Exception as e:
            print(f"   ❌ Ошибка в {file.name}: {e}")
    
    print(f"   ✅ Извлечено: {len(qa_pairs)} Q&A")
    
    # Сохранение
    output_file = DATA_DIR / category / "structured_qa.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"qa_pairs": qa_pairs}, f, ensure_ascii=False, indent=2)
    
    return qa_pairs


def extract_text_from_pdf(file_path):
    """Извлечение текста из PDF"""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except:
        return ""


def extract_text_from_word(file_path):
    """Извлечение текста из DOCX"""
    try:
        doc = docx.Document(file_path)
        text = "\n\n".join([para.text for para in doc.paragraphs])
        return text
    except:
        return ""


def process_instructions_category(category: str):
    """Обработка инструкций одной категории"""
    print(f"\n📚 Обработка инструкций для {category}...")
    
    instr_dir = DATA_DIR / category / "instructions"
    if not instr_dir.exists():
        print(f"   ⚠️ Папка {instr_dir} не найдена")
        return []
    
    pdf_files = list(instr_dir.glob("*.pdf"))
    word_files = list(instr_dir.glob("*.docx"))
    all_files = pdf_files + word_files
    
    print(f"   Найдено файлов: {len(all_files)}")
    
    chunks = []
    
    for file in tqdm(all_files, desc=f"  {category}"):
        try:
            if file.suffix == '.pdf':
                text = extract_text_from_pdf(file)
            else:
                text = extract_text_from_word(file)
            
            if text:
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                for para in paragraphs:
                    if len(para) > 50:
                        chunks.append({
                            "id": f"{category}_chunk_{len(chunks)+1:05d}",
                            "text": para,
                            "source": file.name,
                            "category": category  # ⭐ КАТЕГОРИЯ
                        })
        except Exception as e:
            print(f"   ❌ Ошибка в {file.name}: {e}")
    
    print(f"   ✅ Создано: {len(chunks)} chunks")
    
    # Сохранение
    output_file = DATA_DIR / category / "chunks_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"chunks": chunks}, f, ensure_ascii=False, indent=2)
    
    return chunks


def main():
    """Обработка всех категорий"""
    print("=" * 60)
    print("🚀 ОБРАБОТКА ДАННЫХ ПО КАТЕГОРИЯМ")
    print("=" * 60)
    
    total_qa = 0
    total_chunks = 0
    
    for category in CATEGORIES:
        print(f"\n{'='*60}")
        print(f"📁 КАТЕГОРИЯ: {category.upper()}")
        print(f"{'='*60}")
        
        category_path = DATA_DIR / category
        if not category_path.exists():
            print(f"⚠️ Папка {category} не найдена, пропускаем")
            continue
        
        # Обработка Excel
        qa_pairs = process_excel_category(category)
        total_qa += len(qa_pairs)
        
        # Обработка инструкций
        chunks = process_instructions_category(category)
        total_chunks += len(chunks)
    
    print("\n" + "=" * 60)
    print("✅ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)
    print(f"📊 Всего Q&A: {total_qa}")
    print(f"📚 Всего chunks: {total_chunks}")
    print("=" * 60)
    print("\n💡 Следующий шаг: python src/build_index.py")


if __name__ == "__main__":
    main()