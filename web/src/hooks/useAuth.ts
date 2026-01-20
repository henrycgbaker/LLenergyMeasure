import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import type { User } from '../types'

export function useAuth() {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [searchParams, setSearchParams] = useSearchParams()

  // Check for OAuth callback token in URL
  useEffect(() => {
    const token = searchParams.get('token')
    if (token) {
      api.setToken(token)
      // Remove token from URL
      searchParams.delete('token')
      setSearchParams(searchParams, { replace: true })
    }
  }, [searchParams, setSearchParams])

  // Load current user
  useEffect(() => {
    const loadUser = async () => {
      try {
        const currentUser = await api.getCurrentUser()
        setUser(currentUser)
      } finally {
        setLoading(false)
      }
    }
    loadUser()
  }, [])

  const login = useCallback(() => {
    window.location.href = api.getGitHubAuthUrl()
  }, [])

  const logout = useCallback(() => {
    api.clearToken()
    setUser(null)
  }, [])

  return {
    user,
    loading,
    isAuthenticated: !!user,
    login,
    logout,
  }
}
