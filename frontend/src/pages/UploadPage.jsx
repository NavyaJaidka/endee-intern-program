import React, { useState, useCallback } from 'react'
import { Upload, FileText, Github, AlertCircle, CheckCircle, Loader2 } from 'lucide-react'
import { uploadDocument, uploadGithubRepo } from '../services/api'

const UploadPage = () => {
  const [activeTab, setActiveTab] = useState('file')
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  
  // GitHub state
  const [repoUrl, setRepoUrl] = useState('')
  const [branch, setBranch] = useState('main')

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setTitle(selectedFile.name.replace(/\.[^/.]+$/, ''))
    }
  }

  const handleUpload = async () => {
    if (!file || !title) {
      setMessage({ type: 'error', text: 'Please select a file and enter a title' })
      return
    }

    setLoading(true)
    setMessage(null)
    setUploadProgress(0)

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90))
      }, 200)

      const response = await uploadDocument(file, title)
      
      clearInterval(progressInterval)
      setUploadProgress(100)

      setMessage({
        type: 'success',
        text: `Successfully uploaded! ${response.data.chunks_created} chunks created.`
      })
      
      // Reset form
      setFile(null)
      setTitle('')
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.response?.data?.detail || error.message || 'Upload failed'
      })
    } finally {
      setLoading(false)
      setTimeout(() => setUploadProgress(0), 1000)
    }
  }

  const handleGithubUpload = async () => {
    if (!repoUrl) {
      setMessage({ type: 'error', text: 'Please enter a GitHub repository URL' })
      return
    }

    setLoading(true)
    setMessage(null)

    try {
      const response = await uploadGithubRepo({ repo_url: repoUrl, branch })
      
      setMessage({
        type: 'success',
        text: `Successfully uploaded! ${response.data.chunks_created} chunks created from ${response.data.message.split(' ')[1]} files.`
      })
      
      setRepoUrl('')
      setBranch('main')
    } catch (error) {
      setMessage({
        type: 'error',
        text: error.response?.data?.detail || error.message || 'GitHub upload failed'
      })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto animate-fade-in">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold gradient-text mb-2">Upload Documents</h1>
        <p className="text-slate-400">
          Add documents to your knowledge base using PDF, text files, or GitHub repositories
        </p>
      </div>

      {/* Tabs */}
      <div className="flex mb-6 bg-dark-card rounded-lg p-1">
        <button
          onClick={() => setActiveTab('file')}
          className={`flex-1 flex items-center justify-center space-x-2 py-3 rounded-lg transition-all ${
            activeTab === 'file'
              ? 'bg-primary-600 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <FileText className="w-5 h-5" />
          <span>File Upload</span>
        </button>
        <button
          onClick={() => setActiveTab('github')}
          className={`flex-1 flex items-center justify-center space-x-2 py-3 rounded-lg transition-all ${
            activeTab === 'github'
              ? 'bg-primary-600 text-white'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <Github className="w-5 h-5" />
          <span>GitHub Repository</span>
        </button>
      </div>

      {/* Message */}
      {message && (
        <div className={`mb-6 p-4 rounded-lg flex items-center space-x-2 ${
          message.type === 'error' 
            ? 'bg-red-500/20 text-red-400 border border-red-500/50'
            : 'bg-green-500/20 text-green-400 border border-green-500/50'
        }`}>
          {message.type === 'error' ? (
            <AlertCircle className="w-5 h-5" />
          ) : (
            <CheckCircle className="w-5 h-5" />
          )}
          <span>{message.text}</span>
        </div>
      )}

      {/* File Upload Tab */}
      {activeTab === 'file' && (
        <div className="card space-y-6">
          {/* Drop zone */}
          <div className="border-2 border-dashed border-dark-border rounded-xl p-8 text-center hover:border-primary-500 transition-colors">
            <input
              type="file"
              onChange={handleFileChange}
              accept=".pdf,.txt,.md,.json"
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className="w-12 h-12 mx-auto text-slate-500 mb-4" />
              <p className="text-lg font-medium mb-2">
                {file ? file.name : 'Drop files here or click to browse'}
              </p>
              <p className="text-slate-500 text-sm">
                Supports PDF, TXT, MD, JSON (max 50MB)
              </p>
            </label>
          </div>

          {/* Title input */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Document Title
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Enter document title"
              className="input"
            />
          </div>

          {/* Upload button */}
          <button
            onClick={handleUpload}
            disabled={loading || !file || !title}
            className="w-full btn btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Uploading...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Upload Document</span>
              </>
            )}
          </button>

          {/* Progress bar */}
          {uploadProgress > 0 && (
            <div className="h-2 bg-dark-border rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary-500 to-purple-500 transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          )}
        </div>
      )}

      {/* GitHub Tab */}
      {activeTab === 'github' && (
        <div className="card space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              GitHub Repository URL
            </label>
            <input
              type="text"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              placeholder="https://github.com/owner/repo"
              className="input"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Branch
            </label>
            <input
              type="text"
              value={branch}
              onChange={(e) => setBranch(e.target.value)}
              placeholder="main"
              className="input"
            />
          </div>

          <button
            onClick={handleGithubUpload}
            disabled={loading || !repoUrl}
            className="w-full btn btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing Repository...</span>
              </>
            ) : (
              <>
                <Github className="w-5 h-5" />
                <span>Upload Repository</span>
              </>
            )}
          </button>

          <p className="text-slate-500 text-sm text-center">
            Note: Large repositories may take longer to process
          </p>
        </div>
      )}
    </div>
  )
}

export default UploadPage
