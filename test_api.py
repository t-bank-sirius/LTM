import requests
import time
from typing import Dict, Any, Optional
from datetime import datetime

class LTMClient:
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Ошибка при проверке здоровья API: {e}")
            return False
    
    def store_memory(self, user_id: str, content: str, context: Optional[str] = None) -> Dict[str, Any]:
        data = {
            "user_id": user_id,
            "content": content
        }
        if context:
            data["context"] = context
        
        try:
            response = self.session.post(
                f"{self.base_url}/memory/store",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Ошибка при сохранении памяти: {e}")
            return {}
    
    def search_memory(self, user_id: str, query: str, limit: int = 5, min_score: float = 0.3) -> list:
        data = {
            "user_id": user_id,
            "query": query,
            "limit": limit,
            "min_score": min_score
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/search",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"❌ Ошибка при поиске в памяти: {e}")
            return []

client = LTMClient()

def main():

    
    print("Проверка работоспособности API...")
    if not client.health_check():
        print("❌ API недоступен!")
        return
    print("✅ API работает корректно!")
    
    user_id = "user"
    
    print(f"Сохранение воспоминаний для пользователя '{user_id}'...")
    
    test_memories = [
        {
            "content": "Изучил алгоритм быстрой сортировки и его временную сложность O(n log n)",
            "context": "программирование"
        },
        {
            "content": "Приготовил домашний борщ с говядиной и сметаной",
            "context": "кулинария"
        },
        {
            "content": "Планирую поездку в Японию: Токио, Киото, гора Фудзи",
            "context": "путешествия"
        },
        {
            "content": "Прочитал книгу 'Чистый код' Роберта Мартина",
            "context": "программирование"
        },
        {
            "content": "Научился готовить итальянскую пасту карбонара",
            "context": "кулинария"
        }
    ]
    
    stored_count = 0
    for i, memory in enumerate(test_memories, 1):
        print(f"   📝 {i}/{len(test_memories)}: Сохраняю '{memory['content'][:40]}...'")
        result = client.store_memory(
            user_id=user_id,
            content=memory["content"],
            context=memory["context"]
        )
        if result and result.get("status") == "success":
            stored_count += 1
            print(f"      ✅ ID: {result.get('memory_id', 'неизвестно')}")
        else:
            print(f"      ❌ Ошибка сохранения")
        time.sleep(0.2) 
    
    print(f"\n✅ Успешно сохранено: {stored_count}/{len(test_memories)} воспоминаний")
    
    print(f"Тестирование поиска воспоминаний...")
    
    search_queries = [
        ("алгоритм", "🔍 Поиск по программированию"),
        ("готовить", "🍳 Поиск по кулинарии"),
        ("поездка", "✈️ Поиск по путешествиям"),
        ("книга", "📚 Поиск по книгам")
    ]
    
    all_memories = []
    
    for query, description in search_queries:
        print(f"\n{description}: '{query}'")
        results = client.search_memory(
            user_id=user_id,
            query=query,
            limit=3,
            min_score=0.1
        )
        
        if results:
            print(f"   📊 Найдено результатов: {len(results)}")
            for j, result in enumerate(results, 1):
                content = result.get('content', '')[:50]
                score = result.get('score', 0)
                context = result.get('context', 'Без контекста')
                time_str = result.get('time', 'Время неизвестно')
                print(f"   {j}. Релевантность: {score:.3f}")
                print(f"      📝 Содержание: {content}...")
                print(f"      🏷️ Контекст: {context}")
                print(f"      ⏰ Время: {time_str}")
                all_memories.append(result)
        else:
            print("   ❌ Результаты не найдены")
        
        time.sleep(0.3)
    

if __name__ == "__main__":
    main() 




print(client.store_memory(
    user_id="user",
    content="песик помылся",
    context="песик"
))
print("--------------------------------")
print(client.search_memory(
    user_id="user",
    query="душ",
    limit=10,
    min_score=0.1
))