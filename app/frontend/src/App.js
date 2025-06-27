// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Home from './pages/Home';
import Conversation from './pages/Conversation';
import Chat from './pages/chat';
import Dashboard from './pages/dashboard';
import AppLayout from './pages/AppLayout';

function App() {
  return (
    <Router>
      <Routes>
        {/* Landing Page */}
        <Route path="/" element={<Home />} />

        {/* Existing “analysis” conversation flow */}
        <Route path="/conversation" element={<Conversation />} />

+       {/* New chat stage, separate from Conversation */}
+       <Route path="/chat" element={<Chat />} />

        {/* Dashboard */}
        <Route path="/dashboard" element={<Dashboard />} />

        {/* Combined view */}
        <Route path="/app" element={<AppLayout />} />
      </Routes>
    </Router>
  );
}

export default App;
