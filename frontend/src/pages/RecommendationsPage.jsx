import React, { useState } from 'react'
import { Lightbulb, Loader2, BookOpen, Code, ArrowRight, ExternalLink, CheckCircle } from 'lucide-react'
import { runRecommendationAgent, runAnalysisAgent } from '../services/api'

const RecommendationsPage = () => {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState(null)

  const handleGetRecommendations = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      // Get both recommendations and analysis
      const [recResponse, analysisResponse] = await Promise.all([
        runRecommendationAgent({ query }),
        runAnalysisAgent({ query, top_k: 5 })
      ])

      setRecommendations(recResponse.data)
      setAnalysis(analysisResponse.data)
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to get recommendations')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto animate-fade-in">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold gradient-text mb-2">Recommendations</h1>
        <p className="text-slate-400">
          Get AI-powered recommendations and insights for your research
        </p>
      </div>

      {/* Query input */}
      <div className="card mb-8">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Lightbulb className="absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-500 w-5 h-5" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleGetRecommendations()}
              placeholder="What would you like to learn about?"
              className="input pl-12 pr-4 py-4 text-lg"
            />
          </div>
          <button
            onClick={handleGetRecommendations}
            disabled={loading || !query.trim()}
            className="btn btn-primary px-8 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Lightbulb className="w-5 h-5" />
                <span>Get Recommendations</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/50 text-red-400 p-4 rounded-lg mb-6">
          {error}
        </div>
      )}

      {/* Results */}
      {recommendations && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Recommendations */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
              <Lightbulb className="w-5 h-5 text-yellow-500" />
              <span>Actionable Recommendations</span>
            </h2>

            <div className="space-y-6">
              {/* Suggestions */}
              <div>
                <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center space-x-2">
                  <Code className="w-4 h-4" />
                  <span>Suggestions</span>
                </h3>
                <ul className="space-y-3">
                  {recommendations.result?.suggestions?.map((suggestion, i) => (
                    <li
                      key={i}
                      className="flex items-start space-x-3 bg-dark-bg p-3 rounded-lg"
                    >
                      <ArrowRight className="w-4 h-4 text-primary-500 mt-0.5 flex-shrink-0" />
                      <span className="text-slate-300">{suggestion}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Next Steps */}
              <div>
                <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center space-x-2">
                  <ArrowRight className="w-4 h-4" />
                  <span>Next Steps</span>
                </h3>
                <ul className="space-y-3">
                  {recommendations.result?.next_steps?.map((step, i) => (
                    <li
                      key={i}
                      className="flex items-start space-x-3 bg-dark-bg p-3 rounded-lg"
                    >
                      <span className="w-6 h-6 bg-primary-600/20 text-primary-400 rounded-full flex items-center justify-center text-sm flex-shrink-0">
                        {i + 1}
                      </span>
                      <span className="text-slate-300">{step}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Resources */}
          <div className="card">
            <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
              <BookOpen className="w-5 h-5 text-green-500" />
              <span>Learning Resources</span>
            </h2>

            <div className="space-y-4">
              {recommendations.result?.resources?.map((resource, i) => (
                <div
                  key={i}
                  className="bg-dark-bg p-4 rounded-lg hover:border-primary-500 border border-transparent transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        resource.type === 'documentation' ? 'bg-blue-600/20 text-blue-400' :
                        resource.type === 'tutorial' ? 'bg-green-600/20 text-green-400' :
                        resource.type === 'guide' ? 'bg-purple-600/20 text-purple-400' :
                        'bg-orange-600/20 text-orange-400'
                      }`}>
                        {resource.type}
                      </span>
                    </div>
                    <ExternalLink className="w-4 h-4 text-slate-500" />
                  </div>
                  <h3 className="font-medium mb-1">{resource.title}</h3>
                  <p className="text-sm text-slate-500">{resource.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && analysis.success && (
        <div className="card mt-6">
          <h2 className="text-xl font-semibold mb-6">Document Analysis</h2>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Document Count */}
            <div className="bg-dark-bg p-4 rounded-lg">
              <div className="text-3xl font-bold text-primary-500 mb-1">
                {analysis.result?.document_count || 0}
              </div>
              <div className="text-sm text-slate-400">Documents Analyzed</div>
            </div>

            {/* Sources */}
            <div className="bg-dark-bg p-4 rounded-lg">
              <div className="text-3xl font-bold text-green-500 mb-1">
                {analysis.result?.sources?.length || 0}
              </div>
              <div className="text-sm text-slate-400">Unique Sources</div>
            </div>

            {/* Themes */}
            <div className="bg-dark-bg p-4 rounded-lg">
              <div className="text-3xl font-bold text-purple-500 mb-1">
                {analysis.result?.key_themes?.length || 0}
              </div>
              <div className="text-sm text-slate-400">Key Themes</div>
            </div>
          </div>

          {/* Insights */}
          {analysis.result?.insights?.length > 0 && (
            <div className="mt-6">
              <h3 className="text-sm font-medium text-slate-400 mb-3">Key Insights</h3>
              <div className="flex flex-wrap gap-2">
                {analysis.result.insights.map((insight, i) => (
                  <span
                    key={i}
                    className="px-3 py-1.5 bg-dark-bg rounded-lg text-sm text-slate-300 flex items-center space-x-2"
                  >
                    <CheckCircle className="w-3 h-3 text-green-500" />
                    <span>{insight}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty state */}
      {!loading && !recommendations && !error && (
        <div className="text-center py-12 text-slate-500">
          <Lightbulb className="w-16 h-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Enter a topic to get personalized recommendations</p>
          <p className="text-sm">Based on your documents and research interests</p>
        </div>
      )}
    </div>
  )
}

export default RecommendationsPage
