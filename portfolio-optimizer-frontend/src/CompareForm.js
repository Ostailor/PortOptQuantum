// src/CompareForm.js
import React, { useState } from 'react';
import { Container, Typography, Paper, TextField, Button, Grid, CircularProgress } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import './CompareForm.css'; // <-- Import the new CSS

function CompareForm() {
  const [tickers, setTickers] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("http://localhost:5000/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: tickers.split(",").map((item) => item.trim()),
          start_date: startDate,
          end_date: endDate
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Something went wrong");
      }
      // Navigate to the results page and pass the data along in state
      navigate('/results', { state: { results: data } });
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="compare-form-container">
      <Paper elevation={4} className="compare-paper">
        <Typography variant="h5" gutterBottom align="center">
          Portfolio Optimizer Comparison
        </Typography>
        <form onSubmit={handleSubmit}>
          <TextField
            label="Tickers (comma separated)"
            fullWidth
            margin="normal"
            value={tickers}
            onChange={(e) => setTickers(e.target.value)}
            required
            className="compare-textfield"
          />
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <TextField
                label="Start Date (YYYY-MM-DD)"
                fullWidth
                margin="normal"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                required
              />
            </Grid>
            <Grid item xs={6}>
              <TextField
                label="End Date (YYYY-MM-DD)"
                fullWidth
                margin="normal"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                required
              />
            </Grid>
          </Grid>
          <Button
            variant="contained"
            color="primary"
            type="submit"
            fullWidth
            className="compare-button"
          >
            Compare
          </Button>
        </form>
        {loading && (
          <CircularProgress style={{ display: "block", margin: "1rem auto" }} />
        )}
        {error && (
          <Typography color="error" style={{ marginTop: "1rem" }}>
            {error}
          </Typography>
        )}
      </Paper>
    </div>
  );
}

export default CompareForm;
