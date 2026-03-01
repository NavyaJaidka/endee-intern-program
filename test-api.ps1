# Test API Endpoints for AI Research & Code Copilot

$baseUrl = "http://localhost:8000/api/v1"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Testing AI Research & Code Copilot API" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Test 1: Health Check
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
    Write-Host "Status: $($response.status)" -ForegroundColor Green
    Write-Host "Endee Connected: $($response.endee_connected)" -ForegroundColor Green
    Write-Host "Embedding Model Loaded: $($response.embedding_model_loaded)" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 2: Get Status
Write-Host "Test 2: Get Status" -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "$baseUrl/status" -Method Get
    Write-Host "Version: $($response.version)" -ForegroundColor Green
    Write-Host "Embedding Dimension: $($response.embedding_dimension)" -ForegroundColor Green
    Write-Host "LLM Provider: $($response.llm_provider)" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 3: Upload Text Document
Write-Host "Test 3: Upload Text Document" -ForegroundColor Yellow
$body = '{"title":"Test Document","content":"This is a test document about artificial intelligence and machine learning","source_type":"txt","source_path":"test"}'

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/documents/text" -Method Post -ContentType "application/json" -Body $body
    Write-Host "Success: $($response.success)" -ForegroundColor Green
    Write-Host "Document ID: $($response.doc_id)" -ForegroundColor Green
    Write-Host "Chunks Created: $($response.chunks_created)" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 4: Search Documents
Write-Host "Test 4: Search Documents" -ForegroundColor Yellow
$body = '{"query":"artificial intelligence","top_k":5}'

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/search" -Method Post -ContentType "application/json" -Body $body
    Write-Host "Success: $($response.success)" -ForegroundColor Green
    Write-Host "Total Results: $($response.total)" -ForegroundColor Green
    if ($response.results.Count -gt 0) {
        Write-Host "First Result: $($response.results[0].text)" -ForegroundColor Green
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 5: Generate Embeddings
Write-Host "Test 5: Generate Embeddings" -ForegroundColor Yellow
$body = '["Hello world", "Machine learning is great", "AI research"]'

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/embeddings" -Method Post -ContentType "application/json" -Body $body
    Write-Host "Success: $($response.success)" -ForegroundColor Green
    Write-Host "Dimension: $($response.dimension)" -ForegroundColor Green
    Write-Host "Number of Embeddings: $($response.embeddings.Count)" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 6: Retrieval Agent
Write-Host "Test 6: Retrieval Agent" -ForegroundColor Yellow
$body = '{"query":"artificial intelligence","agent_type":"retrieval","top_k":3}'

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/agents/retrieval" -Method Post -ContentType "application/json" -Body $body
    Write-Host "Success: $($response.success)" -ForegroundColor Green
    Write-Host "Result: $($response.result)" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 7: Recommendation Agent
Write-Host "Test 7: Recommendation Agent" -ForegroundColor Yellow
$body = '{"query":"machine learning"}'

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/agents/recommendation" -Method Post -ContentType "application/json" -Body $body
    Write-Host "Success: $($response.success)" -ForegroundColor Green
    Write-Host "Result: $($response.result)" -ForegroundColor Green
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Tests Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
