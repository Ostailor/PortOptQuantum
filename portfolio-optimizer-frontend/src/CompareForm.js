// src/CompareForm.js
import React, { useState } from 'react';

const CompareForm = () => {
    const [tickers, setTickers] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);
        setResults(null);

        const response = await fetch('http://localhost:5000/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                tickers: tickers.split(',').map(ticker => ticker.trim()),
                start_date: startDate,
                end_date: endDate,
            }),
        });

        if (response.ok) {
            const data = await response.json();
            setResults(data);
        } else {
            const errorData = await response.json();
            setError(errorData.error);
        }
    };

    return (
        <div>
            <h1>Portfolio Optimizer</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>
                        Tickers (comma-separated):
                        <input
                            type="text"
                            value={tickers}
                            onChange={(e) => setTickers(e.target.value)}
                            required
                        />
                    </label>
                </div>
                <div>
                    <label>
                        Start Date (YYYY-MM-DD):
                        <input
                            type="date"
                            value={startDate}
                            onChange={(e) => setStartDate(e.target.value)}
                            required
                        />
                    </label>
                </div>
                <div>
                    <label>
                        End Date (YYYY-MM-DD):
                        <input
                            type="date"
                            value={endDate}
                            onChange={(e) => setEndDate(e.target.value)}
                            required
                        />
                    </label>
                </div>
                <button type="submit">Compare</button>
            </form>
            {error && <p style={{ color: 'red' }}>{error}</p>}
            {results && (
                <div>
                    <h2>Results</h2>
                    <pre>{JSON.stringify(results, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default CompareForm;