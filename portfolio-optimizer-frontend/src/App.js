// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CompareForm from './CompareForm';
import ResultsPage from './ResultsPage';
import './Custom.css'; // You can keep global resets and any extra rules here

const theme = createTheme({
  palette: {
    primary: {
      main: '#61dafb' // Adjust as needed
    },
    secondary: {
      main: '#282c34' // Adjust as needed
    }
  },
  typography: {
    fontFamily: 'Roboto, sans-serif'
  }
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <Router>
        <Routes>
          <Route path="/" element={<CompareForm />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
