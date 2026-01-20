import { useState, useCallback } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { api, getErrorMessage } from '../api/client'
import { useAuth } from '../hooks/useAuth'

interface ParsedResult {
  experiment_id?: string
  model_name?: string
  backend?: string
  total_tokens?: number
  tokens_per_joule?: number
  throughput?: number
}

export default function UploadForm() {
  const navigate = useNavigate()
  const { isAuthenticated, login } = useAuth()
  const [file, setFile] = useState<File | null>(null)
  const [jsonContent, setJsonContent] = useState<Record<string, unknown> | null>(null)
  const [preview, setPreview] = useState<ParsedResult | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [parseError, setParseError] = useState<string | null>(null)

  const uploadMutation = useMutation({
    mutationFn: (data: Record<string, unknown>) => api.uploadBenchmark(data),
    onSuccess: (result) => {
      navigate(`/benchmark/${result.id}`)
    },
  })

  const parseFile = useCallback(async (file: File) => {
    setParseError(null)
    setPreview(null)
    setJsonContent(null)

    try {
      const text = await file.text()
      const json = JSON.parse(text)

      // Validate basic structure
      if (typeof json !== 'object' || json === null) {
        throw new Error('Invalid JSON: expected an object')
      }

      // Extract preview info
      const config = json.effective_config || {}
      const processResults = json.process_results || []
      const modelName = config.model_name || processResults[0]?.model_name || 'Unknown'

      setPreview({
        experiment_id: json.experiment_id,
        model_name: modelName,
        backend: json.backend || config.backend || 'pytorch',
        total_tokens: json.total_tokens,
        tokens_per_joule: json.tokens_per_joule,
        throughput: json.avg_tokens_per_second,
      })

      setJsonContent(json)
      setFile(file)
    } catch (err) {
      if (err instanceof SyntaxError) {
        setParseError('Invalid JSON format')
      } else {
        setParseError(getErrorMessage(err))
      }
      setFile(null)
    }
  }, [])

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      parseFile(e.dataTransfer.files[0])
    }
  }, [parseFile])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      parseFile(e.target.files[0])
    }
  }, [parseFile])

  const handleSubmit = useCallback(() => {
    if (jsonContent) {
      uploadMutation.mutate(jsonContent)
    }
  }, [jsonContent, uploadMutation])

  const formatNumber = (n: number | undefined) => {
    if (n === undefined) return '-'
    if (n >= 1000000) return `${(n / 1000000).toFixed(2)}M`
    if (n >= 1000) return `${(n / 1000).toFixed(2)}K`
    return n.toFixed(2)
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Upload Benchmark Result</h1>
        <p className="text-sm text-gray-600 mt-1">
          Upload the JSON output from your CLI benchmark run
        </p>
      </div>

      {/* Auth notice */}
      {!isAuthenticated && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-start">
            <svg
              className="w-5 h-5 text-yellow-400 mt-0.5"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800">Sign in recommended</h3>
              <p className="text-sm text-yellow-700 mt-1">
                Anonymous uploads are allowed, but signing in lets you track your submissions.
              </p>
              <button
                onClick={login}
                className="mt-2 text-sm font-medium text-yellow-800 hover:text-yellow-900"
              >
                Sign in with GitHub
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Drop zone */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".json,application/json"
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          stroke="currentColor"
          fill="none"
          viewBox="0 0 48 48"
        >
          <path
            d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <p className="mt-4 text-sm text-gray-600">
          <span className="font-medium text-primary-600">Click to upload</span> or drag and drop
        </p>
        <p className="mt-1 text-xs text-gray-500">JSON file (AggregatedResult format)</p>
      </div>

      {/* Parse error */}
      {parseError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm text-red-600">{parseError}</p>
        </div>
      )}

      {/* Preview */}
      {preview && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">Preview</h2>

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500">Model</p>
              <p className="font-medium text-gray-900">{preview.model_name}</p>
            </div>
            <div>
              <p className="text-gray-500">Backend</p>
              <p className="font-medium text-gray-900">{preview.backend}</p>
            </div>
            <div>
              <p className="text-gray-500">Efficiency</p>
              <p className="font-mono font-medium text-gray-900">
                {formatNumber(preview.tokens_per_joule)} tokens/J
              </p>
            </div>
            <div>
              <p className="text-gray-500">Throughput</p>
              <p className="font-mono font-medium text-gray-900">
                {formatNumber(preview.throughput)} tokens/s
              </p>
            </div>
            <div>
              <p className="text-gray-500">Total Tokens</p>
              <p className="font-mono font-medium text-gray-900">
                {formatNumber(preview.total_tokens)}
              </p>
            </div>
            {preview.experiment_id && (
              <div>
                <p className="text-gray-500">Experiment ID</p>
                <p className="font-mono text-gray-900 text-xs truncate">
                  {preview.experiment_id}
                </p>
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-gray-200 flex justify-between items-center">
            <p className="text-sm text-gray-500">
              File: <span className="font-medium">{file?.name}</span>
            </p>
            <button
              onClick={handleSubmit}
              disabled={uploadMutation.isPending}
              className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50"
            >
              {uploadMutation.isPending ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Uploading...
                </>
              ) : (
                'Upload Benchmark'
              )}
            </button>
          </div>

          {/* Upload error */}
          {uploadMutation.isError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 mt-4">
              <p className="text-sm text-red-600">
                {getErrorMessage(uploadMutation.error)}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Help text */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-900 mb-2">How to get benchmark JSON</h3>
        <p className="text-sm text-gray-600">
          Run the CLI tool and find the result JSON in your results directory:
        </p>
        <pre className="mt-2 bg-gray-900 text-gray-100 rounded-lg p-3 text-sm overflow-x-auto">
          llm-energy-measure experiment config.yaml --dataset alpaca -n 100
        </pre>
        <p className="text-sm text-gray-600 mt-2">
          Results are saved to <code className="bg-gray-200 px-1 rounded">results/</code> directory.
        </p>
      </div>
    </div>
  )
}
