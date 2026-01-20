import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Leaderboard from './components/Leaderboard'
import BenchmarkDetail from './components/BenchmarkDetail'
import CompareView from './components/CompareView'
import UploadForm from './components/UploadForm'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Leaderboard />} />
        <Route path="/benchmark/:id" element={<BenchmarkDetail />} />
        <Route path="/compare" element={<CompareView />} />
        <Route path="/upload" element={<UploadForm />} />
      </Routes>
    </Layout>
  )
}

export default App
