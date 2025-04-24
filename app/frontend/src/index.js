import React from 'react';
import ReactDOM from 'react-dom/client';

// 1. Import the global CSS file
import './global_css.css'; // Assuming the file is in the same src directory

import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);