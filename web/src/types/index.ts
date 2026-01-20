// API Types for LLM Bench

export interface User {
  id: number
  github_id: string
  username: string
  email: string | null
  avatar_url: string | null
  created_at: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  user: User
}

export interface BenchmarkSummary {
  id: number
  experiment_id: string
  user_id: number | null
  model_name: string
  model_family: string | null
  backend: string
  hardware: string
  gpu_name: string | null
  tokens_per_joule: number
  throughput_tokens_per_sec: number
  total_energy_joules: number
  avg_energy_per_token_joules: number
  peak_memory_mb: number
  ttft_ms: number | null
  itl_ms: number | null
  total_tokens: number
  input_tokens: number | null
  output_tokens: number | null
  created_at: string
}

export interface BenchmarkResponse extends BenchmarkSummary {
  raw_result: Record<string, unknown>
  config: Record<string, unknown>
  updated_at: string
}

export interface BenchmarkListResponse {
  items: BenchmarkSummary[]
  total: number
  page: number
  per_page: number
  pages: number
}

export interface ComparisonMetric {
  best_id: number
  best_value: number
  values: Record<string, number>
}

export interface CompareResponse {
  benchmarks: BenchmarkResponse[]
  comparison: Record<string, ComparisonMetric>
}

export interface ModelStats {
  model_name: string
  model_family: string | null
  benchmark_count: number
  best_tokens_per_joule: number
  best_throughput: number
  avg_tokens_per_joule: number
  avg_throughput: number
}

export interface ModelsListResponse {
  items: ModelStats[]
  total: number
}

export interface FilterOptions {
  backends: string[]
  hardware: string[]
  model_families: string[]
}

export type SortField =
  | 'tokens_per_joule'
  | 'throughput_tokens_per_sec'
  | 'total_energy_joules'
  | 'peak_memory_mb'
  | 'created_at'

export type SortOrder = 'asc' | 'desc'

export interface ListParams {
  page?: number
  per_page?: number
  sort_by?: SortField
  sort_order?: SortOrder
  backend?: string
  hardware?: string
  model_family?: string
  search?: string
}
