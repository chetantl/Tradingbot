/**
 * Analytics Page
 * Displays comprehensive trading analytics and performance metrics
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Tab,
  Tabs,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
} from '@mui/material';

const Analytics = () => {
  const [tabValue, setTabValue] = useState(0);
  const [analyticsData, setAnalyticsData] = useState({});
  const [signalHistory, setSignalHistory] = useState([]);
  const [performance, setPerformance] = useState({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchAnalyticsData();
  }, []);

  const fetchAnalyticsData = async () => {
    try {
      setIsLoading(true);

      // Fetch statistics
      const statsResponse = await fetch('/api/trading/statistics?days_back=30', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
        },
      });

      if (statsResponse.ok) {
        const stats = await statsResponse.json();
        setAnalyticsData(stats);
        setPerformance({
          winRate: stats.win_rate_percent,
          avgProfit: stats.average_profit_pct,
          totalSignals: stats.total_signals,
        });
      }

      // Fetch signal history
      const historyResponse = await fetch('/api/trading/signals?limit=50', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
        },
      });

      if (historyResponse.ok) {
        const history = await historyResponse.json();
        setSignalHistory(history.signals || []);
      }

    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const getSignalTypeColor = (type) => {
    const colors = {
      ACCUMULATION: '#4CAF50',
      DISTRIBUTION: '#F44336',
      BUY: '#2196F3',
      SELL: '#FF9800',
    };
    return colors[type] || '#9E9E9E';
  };

  const getConfidenceColor = (score) => {
    if (score >= 9) return '#4CAF50';
    if (score >= 7) return '#FF9800';
    return '#F44336';
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Analytics & Performance
      </Typography>

      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Overview" />
          <Tab label="Signal History" />
          <Tab label="Performance" />
        </Tabs>

        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
            <LinearProgress />
          </Box>
        ) : (
          <Box>
            {tabValue === 0 && (
              <Grid container spacing={3}>
                {/* Key Metrics */}
                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Total Signals
                      </Typography>
                      <Typography variant="h3" color="primary">
                        {analyticsData.total_signals || 0}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Win Rate
                      </Typography>
                      <Typography variant="h3" color="success">
                        {performance.winRate ? `${performance.winRate.toFixed(1)}%` : 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Avg Profit
                      </Typography>
                      <Typography variant="h3" color="success">
                        {performance.avgProfit ? `${performance.avgProfit.toFixed(2)}%` : 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                <Grid item xs={12} md={3}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Success Rate
                      </Typography>
                      <Typography variant="h3" color="primary">
                        {performance.winRate ? `${performance.winRate.toFixed(1)}%` : 'N/A'}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>

                {/* Signal Type Distribution */}
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Signal Distribution
                      </Typography>
                      <Box sx={{ mt: 2 }}>
                        {Object.entries(analyticsData.signals_by_type || {}).map(([type, count]) => (
                          <Box key={type} sx={{ mb: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                              <Typography variant="body2">{type}</Typography>
                              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                {count}
                              </Typography>
                            </Box>
                            <LinearProgress
                              variant="determinate"
                              value={(count / (analyticsData.total_signals || 1)) * 100}
                              sx={{
                                height: 6,
                                borderRadius: 3,
                                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                '& .MuiLinearProgress-bar': {
                                  backgroundColor: getSignalTypeColor(type),
                                  borderRadius: 3,
                                },
                              }}
                            />
                          </Box>
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}

            {tabValue === 1 && (
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Signal History
                  </Typography>

                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Time</TableCell>
                          <TableCell>Symbol</TableCell>
                          <TableCell>Type</TableCell>
                          <TableCell>Confidence</TableCell>
                          <TableCell>Entry</TableCell>
                          <TableCell>Target</TableCell>
                          <TableCell>Stop Loss</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {signalHistory.map((signal, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              {new Date(signal.created_time).toLocaleString()}
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                {signal.symbol}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={signal.signal_type}
                                size="small"
                                sx={{
                                  backgroundColor: getSignalTypeColor(signal.signal_type),
                                  color: 'white',
                                }}
                              />
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={`${signal.confidence_score}/10`}
                                size="small"
                                sx={{
                                  backgroundColor: getConfidenceColor(signal.confidence_score),
                                  color: 'white',
                                }}
                              />
                            </TableCell>
                            <TableCell>
                              {signal.entry_price}
                            </TableCell>
                            <TableCell>
                              {signal.target_price}
                            </TableCell>
                            <TableCell>
                              {signal.stop_loss}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>

                  {signalHistory.length === 0 && (
                    <Box sx={{ textAlign: 'center', py: 4 }}>
                      <Typography variant="body2" color="text.secondary">
                        No signal history available
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            )}

            {tabValue === 2 && (
              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Performance Overview
                      </Typography>

                      <Typography variant="body2" paragraph>
                        Detailed performance metrics and trading analytics will be displayed here.
                        Track win rates, average profits, and signal effectiveness over time.
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default Analytics;