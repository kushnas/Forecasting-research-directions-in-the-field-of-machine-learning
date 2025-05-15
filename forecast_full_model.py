# Библиотеки для работы с данными
import pandas as pd
import numpy as np
import ast
import re

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Модели и алгоритмы машинного обучения
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import (train_test_split, TimeSeriesSplit, StratifiedKFold)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, roc_auc_score, 
                             roc_curve)

# Временные ряды
from statsmodels.tsa.arima.model import ARIMA

# Работа с графами
import networkx as nx

# Утилиты
from collections import defaultdict
from tqdm import tqdm
from itertools import combinations
from joblib import Parallel, delayed

# Предупреждения
import warnings
warnings.filterwarnings('ignore')

# Дата и время
from datetime import datetime

warnings.filterwarnings('ignore')

class NLPConceptPredictor:
    def __init__(self, data_path, sample_size=None):
        self.data_path = data_path
        self.sample_size = sample_size
        
    def load_data(self):
        """Безопасная загрузка данных с обработкой ошибок"""
        try:
            print("Загрузка данных...")
            dtype = {
                'work_uri': 'string',
                'publication_year': 'int16',
                'cited_by_count': 'int32'
            }
            usecols = ['work_uri', 'publication_year', 'cited_by_count'] + \
                     [f'collocation_{n}' for n in range(2,5)]
            
            chunks = pd.read_csv(self.data_path, dtype=dtype, usecols=usecols, chunksize=50000)
            data = pd.concat(chunks)
            
            if self.sample_size:
                data = data.sample(min(self.sample_size, len(data)), random_state=42)
            
            # Безопасный парсинг коллокаций
            for n in range(2, 5):
                data[f'collocation_{n}'] = data[f'collocation_{n}'].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) and x.startswith('[') else [])
            
            return data
        except Exception as e:
            print(f"Ошибка загрузки данных: {str(e)}")
            return pd.DataFrame()

    def preprocess_concepts(self, data):
        """Извлечение и нормализация концепций с проверкой данных"""
        if data.empty:
            print("Предупреждение: Нет данных для обработки")
            return set()
            
        print("\nПредобработка концепций...")
        unique_concepts = set()
        
        try:
            for _, row in tqdm(data.iterrows(), total=len(data)):
                # Обрабатываем только коллокации длиной 3
                collocations = row.get('collocation_3', [])
                if isinstance(collocations, list):
                    for concept in collocations:
                        if isinstance(concept, str):
                            normalized = ' '.join(sorted(concept.lower().split()))
                            unique_concepts.add(normalized)
            return unique_concepts
        except Exception as e:
            print(f"Ошибка предобработки концепций: {str(e)}")
            return set()




    # def preprocess_concepts(self, data):
    #     """Извлечение и нормализация концепций с проверкой данных"""
    #     if data.empty:
    #         print("Предупреждение: Нет данных для обработки")
    #         return set()
            
    #     print("\nПредобработка концепций...")
    #     unique_concepts = set()
        
    #     try:
    #         for _, row in tqdm(data.iterrows(), total=len(data)):
    #             for n in range(2, 5):
    #                 collocations = row.get(f'collocation_{n}', [])
    #                 if isinstance(collocations, list):
    #                     for concept in collocations:
    #                         if isinstance(concept, str):
    #                             normalized = ' '.join(sorted(concept.lower().split()))
    #                             unique_concepts.add(normalized)
    #         return unique_concepts
    #     except Exception as e:
    #         print(f"Ошибка предобработки концепций: {str(e)}")
    #         return set()



    def build_temporal_network(self, data, concepts, min_year, max_year):
        """Построение временной сети с обработкой ошибок"""
        if data.empty or not concepts:
            print("Предупреждение: Нет данных или концепций для построения сети")
            return {}, defaultdict(lambda: defaultdict(list))
            
        print("\nПостроение временной сети...")
        yearly_networks = {}
        concept_stats = defaultdict(lambda: defaultdict(list))
        
        try:
            for year in tqdm(range(min_year, max_year)):
                year_data = data[data['publication_year'] == year]
                if year_data.empty:
                    yearly_networks[year] = nx.Graph()
                    continue
                    
                G = nx.Graph()
                concept_cooccurrence = defaultdict(int)
                
                for _, row in year_data.iterrows():
                    current_concepts = set()
                    for n in range(2, 5):
                        collocations = row.get(f'collocation_{n}', [])
                        if isinstance(collocations, list):
                            current_concepts.update(c for c in collocations if c in concepts)
                    
                    for concept in current_concepts:
                        concept_stats[concept]['count'].append(1)
                    
                    for u, v in combinations(current_concepts, 2):
                        concept_cooccurrence[(u, v)] += 1
                
                for (u, v), weight in concept_cooccurrence.items():
                    G.add_edge(u, v, weight=weight)
                
                yearly_networks[year] = G
                
            return yearly_networks, concept_stats
        except Exception as e:
            print(f"Ошибка построения сети: {str(e)}")
            return {}, defaultdict(lambda: defaultdict(list))

    def calculate_concept_features(self, concept, networks, stats, current_year, lookback):
        """Расчет признаков с защитой от ошибок"""
        try:
            features = {}
            years = range(current_year - lookback, current_year)
            counts = []
            degrees = []
            centralities = []
            
            for year in years:
                G = networks.get(year, nx.Graph())
                
                # Частота концепта
                count = sum(stats[concept]['count']) if concept in stats else 0
                counts.append(count)
                
                # Сетевые метрики
                degree = G.degree(concept) if concept in G else 0
                degrees.append(degree)
                
                # Центральность
                try:
                    centrality = nx.betweenness_centrality(G).get(concept, 0)
                    centralities.append(centrality)
                except:
                    centralities.append(0)
            
            if len(counts) < 2:
                return None
                
            # Расчет признаков с защитой от деления на ноль
            total = sum(counts)
            mean_count = np.mean(counts)
            growth_rate = (counts[-1] - counts[0]) / (counts[0] + 1e-9) if counts[0] > 0 else 0
            stability = 1.0 / (np.std(degrees) + 1e-9)
            
            features.update({
                'total_mentions': total,
                'growth_rate': growth_rate,
                'acceleration': (counts[-1] - 2*counts[-2] + counts[-3]) if len(counts) >=3 else 0,
                'stability': stability,
                'mean_centrality': np.mean(centralities),
                'recent_activity': np.mean(counts[-2:]),
                'max_degree': max(degrees) if degrees else 0,
                'yearly_std': np.std(counts),
                'mean_degree': np.mean(degrees)
            })
            
            return features
        except Exception as e:
            print(f"Ошибка расчета признаков для концепта {concept}: {str(e)}")
            return None

    def train_random_forest(self, X, y):
        """Обучение модели с обработкой ошибок и сохранением ROC-AUC кривых"""
        if len(X) == 0 or len(y) == 0:
            print("Ошибка: Нет данных для обучения")
            return None, None
            
        print("\nОбучение Random Forest...")
        try:
            # Используем StratifiedKFold для гарантированного наличия обоих классов
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            train_auc_scores = []
            test_auc_scores = []
            feature_importances = []
            
            plt.figure(figsize=(10, 8))
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Гарантируем наличие обоих классов в тестовой выборке
                while len(np.unique(y_test)) < 2:
                    print("Перемешиваем тестовую выборку для обеспечения обоих классов")
                    np.random.shuffle(test_idx)
                    X_test = X[test_idx]
                    y_test = y[test_idx]
                
                model.fit(X_train, y_train)
                
                # Прогнозы для тренировочной и тестовой выборок
                y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Расчет ROC-AUC
                train_auc = roc_auc_score(y_train, y_train_pred_proba)
                test_auc = roc_auc_score(y_test, y_test_pred_proba)
                
                train_auc_scores.append(train_auc)
                test_auc_scores.append(test_auc)
                
                # Сохранение важности признаков
                feature_importances.append(model.feature_importances_)
                
                # Построение ROC-кривых
                fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
                fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)
                
                plt.plot(fpr_train, tpr_train, linestyle='--', alpha=0.5, 
                         label=f'Train Fold {fold} (AUC = {train_auc:.2f})')
                plt.plot(fpr_test, tpr_test, alpha=0.5, 
                         label=f'Test Fold {fold} (AUC = {test_auc:.2f})')
            
            # Средние значения ROC-AUC
            mean_train_auc = np.mean(train_auc_scores)
            mean_test_auc = np.mean(test_auc_scores)
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves\nMean Train AUC: {mean_train_auc:.3f}, Mean Test AUC: {mean_test_auc:.3f}')
            plt.legend(loc='lower right')
            plt.show()
            
            print(f"\nСредний ROC-AUC на обучении: {mean_train_auc:.3f}")
            print(f"Средний ROC-AUC на тестировании: {mean_test_auc:.3f}")
            
            return model, np.mean(feature_importances, axis=0)
        except Exception as e:
            print(f"Ошибка обучения модели: {str(e)}")
            return None, None

    def get_popular_concepts(self, data, year, top_pct=0.15):
        """Безопасное получение популярных концепций"""
        if data.empty:
            print(f"Предупреждение: Нет данных для {year} года")
            return set()
            
        concept_counts = defaultdict(int)
        year_data = data[data['publication_year'] == year]
        
        if year_data.empty:
            print(f"Предупреждение: Нет данных за {year} год")
            return set()
        
        try:
            for _, row in year_data.iterrows():
                for n in range(2, 5):
                    collocations = row.get(f'collocation_{n}', [])
                    if isinstance(collocations, list):
                        for concept in collocations:
                            if isinstance(concept, str):
                                concept_counts[concept] += 1
            
            if not concept_counts:
                print(f"Предупреждение: Нет концепций для {year} года")
                return set()
            
            counts = list(concept_counts.values())
            if len(counts) == 0:
                return set()
                
            # Альтернативный расчет порога если перцентиль не работает
            try:
                threshold = np.percentile(counts, 100 - top_pct*100)
            except:
                sorted_counts = sorted(counts, reverse=True)
                top_n = max(1, int(len(sorted_counts) * top_pct))
                threshold = sorted_counts[top_n-1] if len(sorted_counts) >= top_n else sorted_counts[-1] if sorted_counts else 0
            
            return {c for c, cnt in concept_counts.items() if cnt >= threshold}
        except Exception as e:
            print(f"Ошибка определения популярных концепций: {str(e)}")
            return set()

    def predict_trends(self, data, predict_year, lookback=5, top_n=50):
        """Прогнозирование трендов с полной обработкой ошибок"""
        if data.empty:
            print("Ошибка: Нет входных данных")
            return [], pd.DataFrame()
            
        print(f"\nПрогнозирование трендов на {predict_year} год...")
        
        try:
            train_data = data[data['publication_year'] < predict_year]
            if train_data.empty:
                print("Ошибка: Нет данных для обучения")
                return [], pd.DataFrame()
                
            concepts = self.preprocess_concepts(train_data)
            if not concepts:
                print("Ошибка: Не удалось извлечь концепции")
                return [], pd.DataFrame()
            
            networks, stats = self.build_temporal_network(
                train_data, concepts, 
                predict_year - lookback, 
                predict_year
            )
            
            features = []
            valid_concepts = []
            
            for concept in tqdm(concepts, desc="Расчет признаков"):
                feat = self.calculate_concept_features(
                    concept, networks, stats, 
                    predict_year, lookback
                )
                if feat and feat['total_mentions'] >= 5:
                    features.append(feat)
                    valid_concepts.append(concept)
            
            if not features:
                print("Ошибка: Не удалось рассчитать признаки")
                return [], pd.DataFrame()
            
            df_features = pd.DataFrame(features, index=valid_concepts)
            if df_features.empty:
                print("Ошибка: Пустой DataFrame признаков")
                return [], pd.DataFrame()
            
            scaler = MinMaxScaler()
            X = scaler.fit_transform(df_features)
            
            popular_concepts = self.get_popular_concepts(train_data, predict_year - 1)
            y = np.array([1 if c in popular_concepts else 0 for c in valid_concepts])
            
            # Гарантируем наличие обоих классов
            if sum(y) < 2:
                print("Предупреждение: Недостаточно положительных примеров. Используем все данные.")
                df_features['score'] = 0
                top_concepts = df_features.sort_values('total_mentions', ascending=False).head(top_n)
                return top_concepts.index.tolist(), df_features
            
            # Балансировка классов для feature selection
            if sum(y) >= 10:
                selector = SelectKBest(mutual_info_classif, k=min(7, X.shape[1]))
                X_selected = selector.fit_transform(X, y)
                selected_features = df_features.columns[selector.get_support()]
                print(f"Отобраны признаки: {list(selected_features)}")
            else:
                X_selected = X
                selected_features = df_features.columns
            
            model, feature_importances = self.train_random_forest(X_selected, y)
            
            if model is None:
                print("Использование резервного метода ранжирования")
                df_features['score'] = df_features['total_mentions'] * df_features['growth_rate']
                top_concepts = df_features.sort_values('score', ascending=False).head(top_n)
            else:
                df_features['score'] = model.predict_proba(X_selected)[:, 1]
                top_concepts = df_features.sort_values('score', ascending=False).head(top_n)
            
            if feature_importances is not None:
                self.plot_feature_importance(selected_features, feature_importances)
            
            return top_concepts.index.tolist(), df_features
        except Exception as e:
            print(f"Критическая ошибка прогнозирования: {str(e)}")
            return [], pd.DataFrame()

    def plot_feature_importance(self, features, importances):
        """Визуализация важности признаков"""
        try:
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Важность признаков в Random Forest')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Ошибка визуализации: {str(e)}")

    def evaluate_predictions(self, predicted, actual):
        """Безопасная оценка предсказаний с ROC-AUC"""
        if not predicted or not actual:
            print("Ошибка: Нет данных для оценки")
            return None, None, None, None
            
        try:
            y_true = [1 if c in actual else 0 for c in predicted]
            
            if len(set(y_true)) < 2:
                print("Предупреждение: Все предсказания одного класса")
                return None, None, None, None
                
            precision = precision_score(y_true, [1]*len(y_true))
            recall = recall_score(y_true, [1]*len(y_true))
            f1 = f1_score(y_true, [1]*len(y_true))
            
            # Для ROC-AUC нужны вероятности, используем индекс в качестве proxy
            y_scores = np.linspace(1, 0, len(y_true))  # Имитируем убывающие вероятности
            auc = roc_auc_score(y_true, y_scores)
            
            print(f"\nPrecision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-score: {f1:.3f}")
            print(f"ROC-AUC: {auc:.3f}")
            
            # Построение ROC-кривой
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.show()
            
            return precision, recall, f1, auc
        except Exception as e:
            print(f"Ошибка оценки: {str(e)}")
            return None, None, None, None

def main():
    # Параметры
    DATA_PATH = 'works_nlp_with_collac.csv'
    SAMPLE_SIZE = 500  #
    PREDICT_YEAR = 2023
    LOOKBACK = 5
    TOP_N = 50
    
    try:
        predictor = NLPConceptPredictor(DATA_PATH, SAMPLE_SIZE)
        data = predictor.load_data()
        
        if data.empty:
            print("Не удалось загрузить данные")
            return
            
        future_concepts, metrics_df = predictor.predict_trends(
            data, PREDICT_YEAR, LOOKBACK, TOP_N
        )
        
        if not future_concepts:
            print("Не удалось получить предсказания")
            return
            
        if PREDICT_YEAR <= max(data['publication_year']):
            test_data = data[data['publication_year'] >= PREDICT_YEAR]
            if not test_data.empty:
                actual_concepts = predictor.get_popular_concepts(test_data, PREDICT_YEAR)
                predictor.evaluate_predictions(future_concepts, actual_concepts)
                
                # Визуализация
                if future_concepts and not metrics_df.empty:
                    plt.figure(figsize=(12, 8))
                    top_concepts = future_concepts[:20]
                    scores = metrics_df.loc[top_concepts, 'score']
                    
                    if 'actual_concepts' in locals():
                        colors = ['green' if c in actual_concepts else 'red' for c in top_concepts]
                    else:
                        colors = 'blue'
                    
                    plt.barh(top_concepts, scores, color=colors)
                    plt.title(f"Топ-20 перспективных концепций на {PREDICT_YEAR}")
                    plt.xlabel("Вероятность актуальности")
                    plt.tight_layout()
                    plt.show()
    except Exception as e:
        print(f"Критическая ошибка в main(): {str(e)}")

if __name__ == "__main__":
    main()