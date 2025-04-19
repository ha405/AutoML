import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';              
import Conversation from './pages/Conversation'; 

function App() {
  return (
    <Router>
      {/* You could add a common Header/Navbar here if needed */}
      <Routes>
        {/* Route for the Home page */}
        <Route path="/" element={<Home />} />

        {/* Route for the Conversation page */}
        <Route path="/conversation" element={<Conversation />} />

        {/* Add other routes here if you expand the app */}
      </Routes>
      {/* You could add a common Footer here if needed */}
    </Router>
  );
}

export default App;