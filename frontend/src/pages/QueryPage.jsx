import React, { useState } from 'react'
import { Search, Send, Loader2, FileText, ExternalLink, Copy, CheckCircle } from 'lucide-react'
import { queryDocuments, runAgentPipeline } from '../services/api'

const QueryPage = () => {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('rag') // 'rag' or 'agent'
  const [copiedIndex, setCopiedIndex] = useState(null)

  const handleQuery = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      let response
      if (mode === 'rag') {
        response = await queryDocuments({ query, top_k: 5 })
      } else {
        response = await runAgentPipeline({ query, top_k: 5 })
      }
      
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Query failed')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleQuery()
    }
  }

  const copyToClipboard = (text, index) => {
    navigator.clipboard.writeText(text)
    setCopiedIndex(index)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  return (
    <div className="max-w-6xl mx-auto animate-fade-in">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold gradient-text mb-2">Ask Questions</h1>
        <p className="text-slate-400">
          Use AI to search and answer questions from your documents
        </p>
      </div>

      {/* Mode selection */}
      <div className="flex justify-center mb-6">
        <div className="bg-dark-card rounded-lg p-1 flex">
          <button
            onClick={() => setMode('rag')}
            className={`px-4 py-2 rounded-lg transition-all ${
              mode === 'rag' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            RAG Query
          </button>
          <button
            onClick={() => setMode('agent')}
            className={`px-4 py-2 rounded-lg transition-all ${
              mode === 'agent' ? 'bg-primary-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            Agent Pipeline
          </button>
        </div>
      </div>

      {/* Query input */}
      <div className="card mb-8">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-500 w-5 h-5" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={mode === 'rag' ? "Ask a question about your documents..." : "Ask the agent pipeline..."}
              className="input pl-12 pr-4 py-4 text-lg"
            />
          </div>
          <button
            onClick={handleQuery}
            disabled={loading || !query.trim()}
            className="btn btn-primary px-8 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Thinking...</span>
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                <span>Ask</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/50 text-red-400 p-4 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Answer */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span>Answer</span>
            </h2>
            <div className="prose prose-invert max-w-none">
              <p className="text-lg leading-relaxed whitespace-pre-wrap">{result.answer}</p>
            </div>
          </div>

          {/* Sources */}
          {result.sources && result.sources.length > 0 && (
            <div className="card">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <FileText className="w-5 h-5 text-primary-500" />
                <span>Sources ({result.sources.length})</span>
              </h2>
              <div className="space-y-4">
                {result.sources.map((source, index) => (
                  <div
                    key={index}
                    className="bg-dark-bg border border-dark-border rounded-lg p-4 hover:border-primary-500 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-primary-400">
                          Source {index + 1}
                        </span>
                        <span className="text-sm text-slate-500">
                          Score: {(source.score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <button
                        onClick={() => copyToClipboard(source.text, index)}
                        className="text-slate-500 hover:text-white transition-colors"
                      >
                        {copiedIndex === index ? (
                          <CheckCircle className="w-4 h-4 text-green-500" />
                        ) : (
                          <Copy className="w-4 h-4" />
                        )}
                      </button>
                    </div>
                    <p className="text-slate-300 text-sm line-clamp-3 mb-2">
                      {source.text}
                    </p>
                    <div className="flex items-center space-x-2 text-xs text-slate-500">
                      <ExternalLink className="w-3 h-3" />
                      <span>{source.source}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Agent Pipeline Results */}
          {mode === 'agent' && result.retrieval && (
            <div className="grid md:grid-cols-2 gap-6">
              {/* Analysis */}
              {result.analysis && (
                <div className="card">
                  <h2 className="text-xl font-semibold mb-4">Analysis</h2>
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-sm font-medium text-slate-400 mb-2">Key Themes</h3>
                      <div className="flex flex-wrap gap-2">
                        {result.analysis.result?.key_themes?.map((theme, i) => (
                          <span
                            key={i}
                            className="px-3 py-1 bg-primary-600/20 text-primary-400 rounded-full text-sm"
                          >
                            {theme}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-slate-400 mb-2">Summary</h3>
                      <p className="text-slate-300 text-sm">
                        {result.analysis.result?.summary}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {result.recommendations && (
                <div className="card">
                  <h2 className="text-xl font-semibold mb-4">Recommendations</h2>
                  <div className="space-y-4">
                    <div>
                      <h3 className="text-sm font-medium text-slate-400 mb-2">Suggestions</h3>
                      <ul className="space-y-2">
                        {result.recommendations.result?.suggestions?.map((suggestion, i) => (
                          <li key={i} className="flex items-start space-x-2 text-sm text-slate-300">
                            <span className="text-primary-500">•</span>
                            <span>{suggestion}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!loading && !result && !error && (
        <div className="text-center py-12 text-slate-500">
          <Search className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Enter a question to get started</p>
          <p className="text-sm">The AI will search your documents and provide an answer</p>
        </div>
      )}
    </div>
  )
}

export default QueryPage
