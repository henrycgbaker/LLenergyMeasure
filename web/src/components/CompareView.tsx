import { useSearchParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { api, getErrorMessage } from '../api/client'
import type { BenchmarkResponse } from '../types'

const COLORS = ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444']

export default function CompareView() {
  const [searchParams] = useSearchParams()
  const idsParam = searchParams.get('ids')
  const ids = idsParam ? idsParam.split(',').map(Number).filter(Boolean) : []

  const { data, isLoading, error } = useQuery({
    queryKey: ['compare', ids],
    queryFn: () => api.compareBenchmarks(ids),
    enabled: ids.length >= 2,
  })

  if (ids.length < 2) {
    return (
      <div className="text-center py-12">
        <h1 className="text-2xl font-bold text-gray-900 mb-4">Compare Benchmarks</h1>
        <p className="text-gray-500 mb-4">
          Select at least 2 benchmarks from the leaderboard to compare.
        </p>
        <Link
          to="/"
          className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700"
        >
          Go to Leaderboard
        </Link>
      </div>
    )
  }

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
      </div>
    )
  }

  if (!data || data.benchmarks.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500">No benchmarks found with the given IDs.</p>
        <Link to="/" className="text-primary-600 hover:underline mt-2 inline-block">
          Back to leaderboard
        </Link>
      </div>
    )
  }

  const { benchmarks, comparison } = data

  // Prepare chart data
  const chartData = [
    {
      metric: 'Efficiency (t/J)',
      ...Object.fromEntries(
        benchmarks.map((b) => [b.model_name, b.tokens_per_joule])
      ),
    },
    {
      metric: 'Throughput (t/s)',
      ...Object.fromEntries(
        benchmarks.map((b) => [b.model_name, b.throughput_tokens_per_sec])
      ),
    },
    {
      metric: 'Memory (MB)',
      ...Object.fromEntries(
        benchmarks.map((b) => [b.model_name, b.peak_memory_mb])
      ),
    },
  ]

  const formatNumber = (n: number) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
    return n.toFixed(2)
  }

  const isBest = (benchmarkId: number, metric: string): boolean => {
    return comparison[metric]?.best_id === benchmarkId
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Compare Benchmarks</h1>
          <p className="text-sm text-gray-600 mt-1">
            Side-by-side comparison of {benchmarks.length} benchmarks
          </p>
        </div>
        <Link
          to="/"
          className="text-sm text-primary-600 hover:text-primary-800"
        >
          Back to leaderboard
        </Link>
      </div>

      {/* Chart */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Metrics Comparison</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" />
              <YAxis />
              <Tooltip />
              <Legend />
              {benchmarks.map((b, i) => (
                <Bar
                  key={b.id}
                  dataKey={b.model_name}
                  fill={COLORS[i % COLORS.length]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Comparison Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                  Metric
                </th>
                {benchmarks.map((b, i) => (
                  <th
                    key={b.id}
                    className="px-4 py-3 text-right text-xs font-medium uppercase"
                    style={{ color: COLORS[i % COLORS.length] }}
                  >
                    {b.model_name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              <ComparisonRow
                label="Efficiency"
                unit="tokens/J"
                benchmarks={benchmarks}
                getValue={(b) => b.tokens_per_joule}
                isBest={(b) => isBest(b.id, 'tokens_per_joule')}
                formatNumber={formatNumber}
              />
              <ComparisonRow
                label="Throughput"
                unit="tokens/s"
                benchmarks={benchmarks}
                getValue={(b) => b.throughput_tokens_per_sec}
                isBest={(b) => isBest(b.id, 'throughput_tokens_per_sec')}
                formatNumber={formatNumber}
              />
              <ComparisonRow
                label="Total Energy"
                unit="J"
                benchmarks={benchmarks}
                getValue={(b) => b.total_energy_joules}
                isBest={(b) => isBest(b.id, 'total_energy_joules')}
                formatNumber={formatNumber}
              />
              <ComparisonRow
                label="Peak Memory"
                unit="MB"
                benchmarks={benchmarks}
                getValue={(b) => b.peak_memory_mb}
                isBest={(b) => isBest(b.id, 'peak_memory_mb')}
                formatNumber={(n) => formatNumber(n)}
              />
              {benchmarks.some((b) => b.ttft_ms) && (
                <ComparisonRow
                  label="TTFT"
                  unit="ms"
                  benchmarks={benchmarks}
                  getValue={(b) => b.ttft_ms}
                  isBest={(b) => isBest(b.id, 'ttft_ms')}
                  formatNumber={(n) => n?.toFixed(2) || '-'}
                />
              )}
              {benchmarks.some((b) => b.itl_ms) && (
                <ComparisonRow
                  label="ITL"
                  unit="ms"
                  benchmarks={benchmarks}
                  getValue={(b) => b.itl_ms}
                  isBest={(b) => isBest(b.id, 'itl_ms')}
                  formatNumber={(n) => n?.toFixed(2) || '-'}
                />
              )}
              <tr className="bg-gray-50">
                <td className="px-4 py-3 text-sm font-medium text-gray-900">Backend</td>
                {benchmarks.map((b) => (
                  <td key={b.id} className="px-4 py-3 text-sm text-right text-gray-700">
                    {b.backend}
                  </td>
                ))}
              </tr>
              <tr>
                <td className="px-4 py-3 text-sm font-medium text-gray-900">Hardware</td>
                {benchmarks.map((b) => (
                  <td key={b.id} className="px-4 py-3 text-sm text-right text-gray-700">
                    {b.hardware}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

interface ComparisonRowProps {
  label: string
  unit: string
  benchmarks: BenchmarkResponse[]
  getValue: (b: BenchmarkResponse) => number | null | undefined
  isBest: (b: BenchmarkResponse) => boolean
  formatNumber: (n: number | null | undefined) => string
}

function ComparisonRow({
  label,
  unit,
  benchmarks,
  getValue,
  isBest,
  formatNumber,
}: ComparisonRowProps) {
  return (
    <tr>
      <td className="px-4 py-3 text-sm font-medium text-gray-900">
        {label}
        <span className="text-xs text-gray-500 ml-1">({unit})</span>
      </td>
      {benchmarks.map((b) => {
        const value = getValue(b)
        const best = isBest(b)
        return (
          <td
            key={b.id}
            className={`px-4 py-3 text-sm text-right font-mono ${
              best ? 'text-green-600 font-bold' : 'text-gray-700'
            }`}
          >
            {value !== null && value !== undefined ? formatNumber(value) : '-'}
            {best && <span className="ml-1 text-xs">*</span>}
          </td>
        )
      })}
    </tr>
  )
}
