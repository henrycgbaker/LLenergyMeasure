import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useSearchParams, useNavigate } from 'react-router-dom'
import { api, getErrorMessage } from '../api/client'
import type { BenchmarkSummary, SortField, SortOrder, ListParams } from '../types'

const SORT_OPTIONS: { value: SortField; label: string }[] = [
  { value: 'tokens_per_joule', label: 'Efficiency (tokens/J)' },
  { value: 'throughput_tokens_per_sec', label: 'Throughput (tokens/s)' },
  { value: 'total_energy_joules', label: 'Energy (J)' },
  { value: 'peak_memory_mb', label: 'Memory (MB)' },
  { value: 'created_at', label: 'Date' },
]

export default function Leaderboard() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set())

  // Parse query params
  const params: ListParams = useMemo(() => ({
    page: parseInt(searchParams.get('page') || '1'),
    per_page: parseInt(searchParams.get('per_page') || '20'),
    sort_by: (searchParams.get('sort_by') as SortField) || 'tokens_per_joule',
    sort_order: (searchParams.get('sort_order') as SortOrder) || 'desc',
    backend: searchParams.get('backend') || undefined,
    hardware: searchParams.get('hardware') || undefined,
    model_family: searchParams.get('model_family') || undefined,
    search: searchParams.get('search') || undefined,
  }), [searchParams])

  // Fetch benchmarks
  const { data, isLoading, error } = useQuery({
    queryKey: ['benchmarks', params],
    queryFn: () => api.listBenchmarks(params),
  })

  // Fetch filter options
  const { data: filters } = useQuery({
    queryKey: ['filters'],
    queryFn: () => api.getFilterOptions(),
  })

  // Update URL params
  const updateParams = (updates: Partial<ListParams>) => {
    const newParams = new URLSearchParams(searchParams)
    Object.entries(updates).forEach(([key, value]) => {
      if (value) {
        newParams.set(key, String(value))
      } else {
        newParams.delete(key)
      }
    })
    // Reset to page 1 when filters change
    if (!updates.page && Object.keys(updates).length > 0) {
      newParams.set('page', '1')
    }
    setSearchParams(newParams)
  }

  // Toggle benchmark selection
  const toggleSelection = (id: number) => {
    const newSelected = new Set(selectedIds)
    if (newSelected.has(id)) {
      newSelected.delete(id)
    } else if (newSelected.size < 4) {
      newSelected.add(id)
    }
    setSelectedIds(newSelected)
  }

  // Go to compare page
  const handleCompare = () => {
    if (selectedIds.size >= 2) {
      const ids = Array.from(selectedIds).join(',')
      navigate(`/compare?ids=${ids}`)
    }
  }

  // Format number with appropriate precision
  const formatNumber = (n: number, decimals = 2) => {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
    return n.toFixed(decimals)
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-600">Error: {getErrorMessage(error)}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Efficiency Leaderboard</h1>
          <p className="text-sm text-gray-600 mt-1">
            Compare LLM inference efficiency across models and hardware
          </p>
        </div>

        {selectedIds.size >= 2 && (
          <button
            onClick={handleCompare}
            className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
          >
            Compare {selectedIds.size} selected
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {/* Search */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Search
            </label>
            <input
              type="text"
              value={params.search || ''}
              onChange={(e) => updateParams({ search: e.target.value || undefined })}
              placeholder="Model name..."
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          {/* Sort by */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Sort by
            </label>
            <select
              value={params.sort_by}
              onChange={(e) => updateParams({ sort_by: e.target.value as SortField })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
            >
              {SORT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Backend filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Backend
            </label>
            <select
              value={params.backend || ''}
              onChange={(e) => updateParams({ backend: e.target.value || undefined })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">All backends</option>
              {filters?.backends.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>

          {/* Hardware filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Hardware
            </label>
            <select
              value={params.hardware || ''}
              onChange={(e) => updateParams({ hardware: e.target.value || undefined })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">All hardware</option>
              {filters?.hardware.map((h) => (
                <option key={h} value={h}>{h}</option>
              ))}
            </select>
          </div>

          {/* Model family filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model Family
            </label>
            <select
              value={params.model_family || ''}
              onChange={(e) => updateParams({ model_family: e.target.value || undefined })}
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="">All families</option>
              {filters?.model_families.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      <span className="sr-only">Select</span>
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Rank
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Model
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Backend
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                      Efficiency
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                      Throughput
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                      Energy
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
                      Memory
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      Hardware
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {data?.items.map((benchmark, index) => (
                    <BenchmarkRow
                      key={benchmark.id}
                      benchmark={benchmark}
                      rank={(params.page! - 1) * params.per_page! + index + 1}
                      selected={selectedIds.has(benchmark.id)}
                      onToggle={() => toggleSelection(benchmark.id)}
                      formatNumber={formatNumber}
                    />
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {data && data.pages > 1 && (
              <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200">
                <div className="text-sm text-gray-500">
                  Showing {(data.page - 1) * data.per_page + 1} to{' '}
                  {Math.min(data.page * data.per_page, data.total)} of {data.total}
                </div>
                <div className="flex space-x-2">
                  <button
                    onClick={() => updateParams({ page: data.page - 1 })}
                    disabled={data.page === 1}
                    className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => updateParams({ page: data.page + 1 })}
                    disabled={data.page === data.pages}
                    className="px-3 py-1 border border-gray-300 rounded text-sm disabled:opacity-50"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

interface BenchmarkRowProps {
  benchmark: BenchmarkSummary
  rank: number
  selected: boolean
  onToggle: () => void
  formatNumber: (n: number, d?: number) => string
}

function BenchmarkRow({ benchmark, rank, selected, onToggle, formatNumber }: BenchmarkRowProps) {
  return (
    <tr className={`hover:bg-gray-50 ${selected ? 'bg-primary-50' : ''}`}>
      <td className="px-4 py-3">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggle}
          className="h-4 w-4 text-primary-600 rounded border-gray-300 focus:ring-primary-500"
        />
      </td>
      <td className="px-4 py-3 text-sm font-medium text-gray-900">
        #{rank}
      </td>
      <td className="px-4 py-3">
        <Link
          to={`/benchmark/${benchmark.id}`}
          className="text-sm font-medium text-primary-600 hover:text-primary-800"
        >
          {benchmark.model_name}
        </Link>
        {benchmark.model_family && (
          <span className="ml-2 text-xs text-gray-500">
            {benchmark.model_family}
          </span>
        )}
      </td>
      <td className="px-4 py-3 text-sm text-gray-500">
        {benchmark.backend}
      </td>
      <td className="px-4 py-3 text-sm text-right font-mono text-gray-900">
        {formatNumber(benchmark.tokens_per_joule)} t/J
      </td>
      <td className="px-4 py-3 text-sm text-right font-mono text-gray-900">
        {formatNumber(benchmark.throughput_tokens_per_sec)} t/s
      </td>
      <td className="px-4 py-3 text-sm text-right font-mono text-gray-500">
        {formatNumber(benchmark.total_energy_joules)} J
      </td>
      <td className="px-4 py-3 text-sm text-right font-mono text-gray-500">
        {formatNumber(benchmark.peak_memory_mb, 0)} MB
      </td>
      <td className="px-4 py-3 text-sm text-gray-500 truncate max-w-[150px]" title={benchmark.hardware}>
        {benchmark.hardware}
      </td>
    </tr>
  )
}
