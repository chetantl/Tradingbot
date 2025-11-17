/**
 * Professional Trading Dashboard
 * Main dashboard view with real-time signals and market data
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Alert,
  Fade,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  NotificationsActive,
  Assessment,
  Speed,
} from '@mui/icons-material';

import { useWebSocket } from '../contexts/WebSocketContext';
import { useNotifications } from '../contexts/NotificationContext';
import { useAuth } from '../contexts/AuthContext';

import SignalCard from '../components/Cards/SignalCard';
import MetricsCards from '../components/Metrics/MetricsCards';
import SignalChart from '../components/Charts/SignalChart';
import RecentActivity from '../components/Activity/RecentActivity';
import MarketOverview from '../components/Market/MarketOverview';

const Dashboard = () => {
  const { isConnected, connectionStatus } = useWebSocket();
  const { showConnectionStatus } = useNotifications();
  const { user } = useAuth();

  const [signals, setSignals] = useState([]);
  const [marketData, setMarketData] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch initial data
  const fetchDashboardData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // Fetch signals
      const signalsResponse = await fetch('/api/trading/signals?limit=10', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
        },
      });

      if (!signalsResponse.ok) {
        throw new Error('Failed to fetch signals');
      }

      const signalsData = await signalsResponse.json();
      setSignals(signalsData.signals || []);

      // Fetch market status
      const statusResponse = await fetch('/api/trading/status', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
        },
      });

      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        setMarketData(statusData);
      }

    } catch (err) {
      console.error('Failed to fetch dashboard data:', err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle new signals from WebSocket
  const handleNewSignal = useCallback((event) => {
    const signal = event.detail;
    setSignals(prev => [signal, ...prev.slice(0, 9)]); // Keep only top 10
  }, []);

  // Handle connection status changes
  useEffect(() => {
    if (connectionStatus !== 'connected') {
      showConnectionStatus(connectionStatus);
    }
  }, [connectionStatus, showConnectionStatus]);

  // Initial data fetch
  useEffect(() => {
    fetchDashboardData();

    // Set up WebSocket event listeners
    window.addEventListener('newSignal', handleNewSignal);

    // Set up periodic data refresh
    const refreshInterval = setInterval(fetchDashboardData, 60000); // Every minute

    return () => {
      window.removeEventListener('newSignal', handleNewSignal);
      clearInterval(refreshInterval);
    };
  }, [fetchDashboardData, handleNewSignal]);

  // Manual refresh
  const handleRefresh = () => {
    fetchDashboardData();
  };

  const getConnectionStatusChip = () => {
    const statusConfig = {
      connected: { color: 'success', label: 'Connected', icon: <Speed /> },
      disconnected: { color: 'error', label: 'Disconnected', icon: <NotificationsActive /> },
      reconnecting: { color: 'warning', label: 'Reconnecting...', icon: <Timeline /> },
      error: { color: 'error', label: 'Connection Error', icon: <Timeline /> },
    };

    const config = statusConfig[connectionStatus] || statusConfig.disconnected;

    return (
      <Chip
        icon={config.icon}
        label={config.label}
        color={config.color}
        variant="outlined"
        size="small"
      />
    );
  };

  if (error && !signals.length) {
    return (
      <Box p={3}>
        <Alert severity="error" onClose={() => setError(null)}>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Trading Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Welcome back, {user?.user_name || 'Trader'}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          {getConnectionStatusChip()}
          {signals.length > 0 && (
            <Chip
              label={`${signals.length} Active Signals`}
              color="primary"
              variant="filled"
            />
          )}
        </Box>
      </Box>

      {/* Loading State */}
      {isLoading && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
        </Box>
      )}

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Metrics Cards Row */}
        <Grid item xs={12}>
          <MetricsCards marketData={marketData} />
        </Grid>

        {/* Left Column - Signals and Market Overview */}
        <Grid item xs={12} lg={8}>
          <Grid container spacing={3}>
            {/* Active Signals */}
            <Grid item xs={12}>
              <Fade in={!isLoading}>
                <Card sx={{ height: 600 }}>
                  <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <TrendingUp color="primary" />
                        Active Signals
                      </Typography>
                      <Chip
                        label={`Last updated: ${new Date().toLocaleTimeString()}`}
                        size="small"
                        variant="outlined"
                        onClick={handleRefresh}
                        sx={{ cursor: 'pointer' }}
                      />
                    </Box>

                    {signals.length === 0 && !isLoading ? (
                      <Box
                        sx={{
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          height: 400,
                          color: 'text.secondary',
                        }}
                      >
                        <Assessment sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                        <Typography variant="h6" gutterBottom>
                          No Active Signals
                        </Typography>
                        <Typography variant="body2" align="center">
                          High-quality trading signals will appear here when market conditions are favorable.
                          <br />
                          Signals require a minimum confidence score of 7/10.
                        </Typography>
                      </Box>
                    ) : (
                      <Box sx={{ maxHeight: 500, overflowY: 'auto' }}>
                        {signals.map((signal, index) => (
                          <SignalCard
                            key={signal.id || index}
                            signal={signal}
                            index={index}
                          />
                        ))}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Fade>
            </Grid>

            {/* Signal Chart */}
            <Grid item xs={12}>
              <SignalChart signals={signals} />
            </Grid>
          </Grid>
        </Grid>

        {/* Right Column - Activity and Market Data */}
        <Grid item xs={12} lg={4}>
          <Grid container spacing={3}>
            {/* Market Overview */}
            <Grid item xs={12}>
              <MarketOverview isConnected={isConnected} />
            </Grid>

            {/* Recent Activity */}
            <Grid item xs={12}>
              <RecentActivity />
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Connection Status Alert */}
      {!isConnected && (
        <Box sx={{ mt: 3 }}>
          <Alert
            severity="warning"
            action={
              <Chip
                label="Reconnect"
                size="small"
                onClick={handleRefresh}
                sx={{ cursor: 'pointer' }}
              />
            }
          >
            Real-time data connection is lost. Some features may not update until connection is restored.
          </Alert>
        </Box>
      )}
    </Box>
  );
};

export default Dashboard;