import pandas as pd
from keybert import KeyBERT
import logging
import time

# Настройка логирования
logging.basicConfig(filename='processing_log5.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Загрузка данных, пропуская первые 200000 строк и загружая следующие 200000
work_abstracts = pd.read_csv('abstracts.csv', skiprows=range(1, 800000), nrows=200000)



# Инициализация модели KeyBERT
kw_model = KeyBERT()

# Начало отсчета времени
start_time = time.time()

# Извлечение ключевых слов
work_abstracts['keywords'] = work_abstracts['abstract'].apply(
    lambda x: kw_model.extract_keywords(x, top_n=50) if x is not None and isinstance(x, str) else []
)

# Сохранение DataFrame в CSV файл
work_abstracts.to_csv('works_abstracts_with_keywords_800000_to_end.csv', index=False)

# Завершение отсчета времени
end_time = time.time()
execution_time = end_time - start_time

# Логирование результатов
logging.info(f'Processed {len(work_abstracts)} rows.')
logging.info(f'Execution time: {execution_time:.2f} seconds.')

print('all done!')