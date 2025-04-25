import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';              
import Conversation from './pages/Conversation'; 

function App() {
  return (
    <Router>
      {}
      <Routes>
        {}
        <Route path="/" element={<Home />} />

        {}
        <Route path="/conversation" element={<Conversation />} />

        {}
      </Routes>
      {}
    </Router>
  );
}

export default App;