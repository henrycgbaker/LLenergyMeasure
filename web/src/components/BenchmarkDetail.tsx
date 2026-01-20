import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { api, getErrorMessage } from '../api/client'

export default function BenchmarkDetail() {
  const { id } = useParams<{ id: string }>()

  const { data: benchmark, isLoading, error } = useQuery({
    queryKey: ['benchmark', id],
    queryFn: () => api.getBenchmark(parseInt(id!)),
    enabled: !!id,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">Error: {getErrorMessage(error)}</p>
        <Link to="/" className="text-primary-600 hover:underline mt-2 inline-block">
          Back to leaderboard
        </Link>
      </div>
    )
  }

  if (!benchmark) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">Benchmark not found</p>
        <Link to="/" className="text-primary-600 hover:underline mt-2 inline-block">
          Back to leaderboard
        </Link>
      </div>
    )
  }

  const formatNumber = (n: number, decimals = 2) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(decimals)}M`
    if (n >= 1000) return `${(n / 1000).toFixed(decimals)}K`
    return n.toFixed(decimals)
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString()
  }

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <nav className="text-sm">
        <Link to="/" className="text-gray-500 hover:text-gray-700">
          Leaderboard
        </Link>
        <span className="mx-2 text-gray-400">/</span>
        <span className="text-gray-900">{benchmark.model_name}</span>
      </nav>

      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {benchmark.model_name}
            </h1>
            <div className="flex flex-wrap gap-2 mt-2">
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800">
                {benchmark.backend}
              </span>
              {benchmark.model_family && (
                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                  {benchmark.model_family}
                </span>
              )}
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                const blob = new Blob([JSON.stringify(benchmark.raw_result, null, 2)], {
                  type: 'application/json',
                })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `benchmark-${benchmark.experiment_id}.json`
                a.click()
                URL.revokeObjectURL(url)
              }}
              className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
            >
              Download JSON
            </button>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Efficiency"
          value={formatNumber(benchmark.tokens_per_joule)}
          unit="tokens/J"
          highlight
        />
        <MetricCard
          label="Throughput"
          value={formatNumber(benchmark.throughput_tokens_per_sec)}
          unit="tokens/s"
        />
        <MetricCard
          label="Total Energy"
          value={formatNumber(benchmark.total_energy_joules)}
          unit="Joules"
        />
        <MetricCard
          label="Peak Memory"
          value={formatNumber(benchmark.peak_memory_mb, 0)}
          unit="MB"
        />
      </div>

      {/* Latency Metrics (if available) */}
      {(benchmark.ttft_ms || benchmark.itl_ms) && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Latency Metrics</h2>
          <div className="grid grid-cols-2 gap-4">
            {benchmark.ttft_ms && (
              <div>
                <p className="text-sm text-gray-500">Time to First Token (TTFT)</p>
                <p className="text-lg font-mono text-gray-900">
                  {benchmark.ttft_ms.toFixed(2)} ms
                </p>
              </div>
            )}
            {benchmark.itl_ms && (
              <div>
                <p className="text-sm text-gray-500">Inter-Token Latency (ITL)</p>
                <p className="text-lg font-mono text-gray-900">
                  {benchmark.itl_ms.toFixed(2)} ms
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Details Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Hardware & Config */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Hardware & Configuration</h2>
          <dl className="space-y-3">
            <DetailRow label="Hardware" value={benchmark.hardware} />
            {benchmark.gpu_name && (
              <DetailRow label="GPU" value={benchmark.gpu_name} />
            )}
            <DetailRow label="Backend" value={benchmark.backend} />
            <DetailRow label="Total Tokens" value={formatNumber(benchmark.total_tokens, 0)} />
            {benchmark.input_tokens && (
              <DetailRow label="Input Tokens" value={formatNumber(benchmark.input_tokens, 0)} />
            )}
            {benchmark.output_tokens && (
              <DetailRow label="Output Tokens" value={formatNumber(benchmark.output_tokens, 0)} />
            )}
          </dl>
        </div>

        {/* Timestamps */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Metadata</h2>
          <dl className="space-y-3">
            <DetailRow label="Experiment ID" value={benchmark.experiment_id} mono />
            <DetailRow label="Created" value={formatDate(benchmark.created_at)} />
            <DetailRow label="Updated" value={formatDate(benchmark.updated_at)} />
            {benchmark.user_id && (
              <DetailRow label="Uploaded by" value={`User #${benchmark.user_id}`} />
            )}
          </dl>
        </div>
      </div>

      {/* Configuration JSON */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Experiment Configuration</h2>
        <pre className="bg-gray-50 rounded-lg p-4 overflow-x-auto text-sm text-gray-800">
          {JSON.stringify(benchmark.config, null, 2)}
        </pre>
      </div>
    </div>
  )
}

interface MetricCardProps {
  label: string
  value: string
  unit: string
  highlight?: boolean
}

function MetricCard({ label, value, unit, highlight }: MetricCardProps) {
  return (
    <div
      className={`rounded-lg p-4 ${
        highlight
          ? 'bg-primary-50 border border-primary-200'
          : 'bg-white border border-gray-200 shadow-sm'
      }`}
    >
      <p className="text-sm text-gray-500">{label}</p>
      <p className={`text-2xl font-bold font-mono ${highlight ? 'text-primary-700' : 'text-gray-900'}`}>
        {value}
      </p>
      <p className="text-sm text-gray-400">{unit}</p>
    </div>
  )
}

interface DetailRowProps {
  label: string
  value: string
  mono?: boolean
}

function DetailRow({ label, value, mono }: DetailRowProps) {
  return (
    <div className="flex justify-between">
      <dt className="text-sm text-gray-500">{label}</dt>
      <dd className={`text-sm text-gray-900 ${mono ? 'font-mono' : ''}`}>{value}</dd>
    </div>
  )
}
