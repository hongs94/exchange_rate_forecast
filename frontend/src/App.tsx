import { Routes, Route } from "react-router-dom"
import Dashboard from "@/pages/dashboard/page"
import Login from "@/pages/Login"
import Signup from "@/pages/Signup"
import "@/index.css"

function App() {
  return (
    <Routes>
      <Route path="/" element={<Dashboard />} />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
    </Routes>
  )
}

export default App