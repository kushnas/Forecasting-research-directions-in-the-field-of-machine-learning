import requests

def get_concepts(work_uri):
    # Формируем URL для получения данных по конкретному work_uri
    url = f"https://api.openalex.org/works"
    
    # Выполняем GET запрос
    response = requests.get(url)
    
    # Проверяем на успешный статус код
    if response.status_code == 200:
        data = response.json()
        # Извлекаем концепции (keywords)
        concepts = data.get('concepts', [])
        
        # Извлекаем только названия концепций
        concept_names = [concept['display_name'] for concept in concepts]
        return concept_names
    else:
        print(f"Error: Unable to fetch data for work_uri {work_uri}. Status code: {response.status_code}")
        return []

# Использование функции
work_uri = "https://doi.org/10.1234/example"  # замените на ваш work_uri
concepts = get_concepts(work_uri)

print("Концепции:", concepts)

work_uris = [
    "https://semopenalex.org/work/W2357953609"
    # Добавьте другие work_uri здесь
]

all_concepts = {}

for uri in work_uris:
    concepts = get_concepts(uri)
    all_concepts[uri] = concepts

# Выводим полученные концепции
for uri, concepts in all_concepts.items():
    print(f"{uri}: {concepts}")