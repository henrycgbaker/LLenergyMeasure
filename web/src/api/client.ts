import axios, { AxiosError, AxiosInstance } from 'axios'
import type {
  BenchmarkListResponse,
  BenchmarkResponse,
  CompareResponse,
  FilterOptions,
  ListParams,
  ModelsListResponse,
  TokenResponse,
  User,
} from '../types'

const API_BASE = import.meta.env.VITE_API_URL || ''

class ApiClient {
  private client: AxiosInstance
  private token: string | null = null

  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE}/api`,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Load token from localStorage
    this.token = localStorage.getItem('auth_token')

    // Add auth header interceptor
    this.client.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`
      }
      return config
    })
  }

  // Auth methods
  setToken(token: string) {
    this.token = token
    localStorage.setItem('auth_token', token)
  }

  clearToken() {
    this.token = null
    localStorage.removeItem('auth_token')
  }

  getGitHubAuthUrl(): string {
    return `${API_BASE}/api/auth/github`
  }

  async getCurrentUser(): Promise<User | null> {
    if (!this.token) return null
    try {
      const response = await this.client.get<User>('/auth/me')
      return response.data
    } catch {
      this.clearToken()
      return null
    }
  }

  // Benchmark methods
  async listBenchmarks(params: ListParams = {}): Promise<BenchmarkListResponse> {
    const response = await this.client.get<BenchmarkListResponse>('/benchmarks', {
      params,
    })
    return response.data
  }

  async getBenchmark(id: number): Promise<BenchmarkResponse> {
    const response = await this.client.get<BenchmarkResponse>(`/benchmarks/${id}`)
    return response.data
  }

  async compareBenchmarks(ids: number[]): Promise<CompareResponse> {
    const response = await this.client.get<CompareResponse>('/benchmarks/compare', {
      params: { ids },
      paramsSerializer: { indexes: null },
    })
    return response.data
  }

  async uploadBenchmark(rawResult: Record<string, unknown>): Promise<BenchmarkResponse> {
    const response = await this.client.post<BenchmarkResponse>('/benchmarks', {
      raw_result: rawResult,
    })
    return response.data
  }

  async getFilterOptions(): Promise<FilterOptions> {
    const response = await this.client.get<FilterOptions>('/benchmarks/filters')
    return response.data
  }

  // Models methods
  async listModels(): Promise<ModelsListResponse> {
    const response = await this.client.get<ModelsListResponse>('/models')
    return response.data
  }
}

export const api = new ApiClient()

// Error handling helper
export function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    return error.response?.data?.detail || error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unknown error occurred'
}
