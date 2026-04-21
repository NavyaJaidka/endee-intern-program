import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Health & Status
export const checkHealth = () => api.get('/health')
export const getStatus = () => api.get('/status')

// Index Management
export const createIndex = (data) => api.post('/index/create', data)
export const listIndexes = () => api.get('/index/list')
export const deleteIndex = (indexName) => api.delete(`/index/${indexName}`)

// Document Upload
export const uploadDocument = async (file, title, indexName) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('title', title)
  if (indexName) formData.append('index_name', indexName)
  
  return api.post('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const uploadTextDocument = (data) => api.post('/documents/text', data)
export const uploadGithubRepo = (data) => api.post('/documents/github', data)

// Search
export const searchDocuments = (data) => api.post('/search', data)

// Query (RAG)
export const queryDocuments = (data) => api.post('/query', data)

// Agents
export const runRetrievalAgent = (data) => api.post('/agents/retrieval', data)
export const runAnalysisAgent = (data) => api.post('/agents/analysis', data)
export const runRecommendationAgent = (data) => api.post('/agents/recommendation', data)
export const runAgentPipeline = (data) => api.post('/agents/pipeline', data)

// Embeddings
export const generateEmbeddings = (texts) => api.post('/embeddings', texts)

export default api
