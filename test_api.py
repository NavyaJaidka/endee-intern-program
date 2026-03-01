"""Test API Endpoints for AI Research & Code Copilot"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_health():
    """Test 1: Health Check"""
    print("\n=== Test 1: Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_status():
    """Test 2: Get Status"""
    print("\n=== Test 2: Get Status ===")
    response = requests.get(f"{BASE_URL}/status")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_upload_text():
    """Test 3: Upload Text Document"""
    print("\n=== Test 3: Upload Text Document ===")
    data = {
        "title": "Test Document",
        "content": "This is a test document about artificial intelligence and machine learning. AI is transforming the world.",
        "source_type": "txt",
        "source_path": "test"
    }
    response = requests.post(f"{BASE_URL}/documents/text", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_search():
    """Test 4: Search Documents"""
    print("\n=== Test 4: Search Documents ===")
    data = {
        "query": "artificial intelligence",
        "top_k": 5
    }
    response = requests.post(f"{BASE_URL}/search", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_embeddings():
    """Test 5: Generate Embeddings"""
    print("\n=== Test 5: Generate Embeddings ===")
    data = ["Hello world", "Machine learning is great", "AI research"]
    response = requests.post(f"{BASE_URL}/embeddings", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_retrieval_agent():
    """Test 6: Retrieval Agent"""
    print("\n=== Test 6: Retrieval Agent ===")
    data = {
        "query": "artificial intelligence",
        "agent_type": "retrieval",
        "top_k": 3
    }
    response = requests.post(f"{BASE_URL}/agents/retrieval", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

def test_recommendation_agent():
    """Test 7: Recommendation Agent"""
    print("\n=== Test 7: Recommendation Agent ===")
    data = {"query": "machine learning"}
    response = requests.post(f"{BASE_URL}/agents/recommendation", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.json()

if __name__ == "__main__":
    print("=" * 50)
    print("Testing AI Research & Code Copilot API")
    print("=" * 50)
    
    try:
        test_health()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        test_status()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        test_upload_text()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        test_search()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        test_embeddings()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        test_retrieval_agent()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        test_recommendation_agent()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("API Tests Complete!")
    print("=" * 50)
