// src/ResultsPage.js
import React from 'react';
import { Container, Paper, Typography, Grid, Table, TableBody, TableCell, TableRow, Divider, Button } from '@mui/material';
import { useLocation, useNavigate } from 'react-router-dom';
import './ResultsPage.css';  // Import the custom results styling

function ResultsPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { results } = location.state || {};

  // Redirect to the form if no results are available.
  if (!results) {
    navigate("/");
    return null;
  }

  // Destructure results data
  const { classical_solution: classical, quantum_solution: quantum, differences } = results;

  return (
    <div className="results-page-container">
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        <Paper elevation={4} className="results-paper">
          <Typography variant="h4" gutterBottom align="center">
            Comparison Results
          </Typography>

          {/* Container for the two solution cards */}
          <Grid container spacing={4} justifyContent="center">
            {/* Classical Solution Section */}
            <Grid item xs={12} md={5}>
              <Paper variant="outlined" sx={{ p: 2, backgroundColor: '#f9f9f9' }}>
                <Typography variant="h5" gutterBottom align="center" color="primary">
                  Classical Solution
                </Typography>
                <Table>
                  <TableBody>
                    <TableRow>
                      <TableCell><strong>Expected Return</strong></TableCell>
                      <TableCell>{classical.expected_return}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><strong>Sharpe Ratio</strong></TableCell>
                      <TableCell>{classical.sharpe_ratio}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><strong>Volatility</strong></TableCell>
                      <TableCell>{classical.volatility}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell colSpan={2}>
                        <Typography variant="subtitle1" sx={{ mb: 1 }}><strong>Weights</strong></Typography>
                        <Table size="small">
                          <TableBody>
                            {Object.entries(classical.weights).map(([ticker, weight]) => (
                              <TableRow key={ticker}>
                                <TableCell>{ticker}</TableCell>
                                <TableCell>{weight}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </Paper>
            </Grid>

            {/* Quantum Solution Section */}
            <Grid item xs={12} md={5}>
              <Paper variant="outlined" sx={{ p: 2, backgroundColor: '#f9f9f9' }}>
                <Typography variant="h5" gutterBottom align="center" color="secondary">
                  Quantum Solution
                </Typography>
                <Table>
                  <TableBody>
                    <TableRow>
                      <TableCell><strong>Expected Return</strong></TableCell>
                      <TableCell>{quantum.expected_return}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><strong>Sharpe Ratio</strong></TableCell>
                      <TableCell>{quantum.sharpe_ratio}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell><strong>Volatility</strong></TableCell>
                      <TableCell>{quantum.volatility}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell colSpan={2}>
                        <Typography variant="subtitle1" sx={{ mb: 1 }}><strong>Weights</strong></Typography>
                        <Table size="small">
                          <TableBody>
                            {Object.entries(quantum.weights).map(([ticker, weight]) => (
                              <TableRow key={ticker}>
                                <TableCell>{ticker}</TableCell>
                                <TableCell>{weight}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </Paper>
            </Grid>
          </Grid>

          <Divider sx={{ my: 4 }} />

          {/* Differences Section */}
          <Typography variant="h5" align="center" gutterBottom>
            Differences
          </Typography>
          <Grid container spacing={2} justifyContent="center">
            <Grid item xs={12} sm={4}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle1"><strong>Return Difference:</strong></Typography>
                <Typography variant="body1">{differences.return_difference}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle1"><strong>Volatility Difference:</strong></Typography>
                <Typography variant="body1">{differences.volatility_difference}</Typography>
              </Paper>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle1"><strong>Weight Difference:</strong></Typography>
                <Typography variant="body1">{differences.weight_difference}</Typography>
              </Paper>
            </Grid>
          </Grid>

          <Button
            variant="contained"
            color="primary"
            sx={{ mt: 4, display: 'block', mx: 'auto' }}
            onClick={() => navigate("/")}
          >
            Back to Form
          </Button>
        </Paper>
      </Container>
    </div>
  );
}

export default ResultsPage;
