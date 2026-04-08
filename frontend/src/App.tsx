// ================================
// FILE STATUS: FROZEN
// Verified working with backend
// Do NOT modify unless API changes
// ================================

import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import InputPage from "./pages/InputPage";
import Result from "./pages/Result";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/input" element={<InputPage />} />
      <Route path="/result" element={<Result />} />
    </Routes>
  );
}

