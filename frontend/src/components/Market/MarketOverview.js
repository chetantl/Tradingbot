/**
 * Market Overview Component
 * Displays market status and key information
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  List,
  ListItem,
  ListItemText,
  LinearProgress,
} from '@mui/material';
import {
  Timeline,
  Speed,
  TrendingUp,
  TrendingDown,
  Assessment,
  NotificationsActive,
} from '@mui/icons-material';

const MarketOverview = ({ isConnected }) => {
  const [marketStatus, setMarketStatus] = useState({});
  const [indices, setIndices] = useState([]);

  useEffect(() => {
    // Mock market data
    setMarketStatus({
      status: isConnected ? 'Open' : 'Closed',
      nextOpen: isConnected ? null : '9:15 AM',
      timeRemaining: isConnected ? null : '5h 23m',
    });

    setIndices([
      { name: 'NIFTY', value: 19845.30, change: 125.50, changePercent: 0.64 },
      { name: 'SENSEX', value: 65732.89, change: 398.15, changePercent: 0.61 },
      { name: 'BANK NIFTY', value: 44123.50, change: 234.75, changePercent: 0.53 },
    ]);
  }, [isConnected]);

  const getStatusColor = (status) => {
    switch (status) {
      case 'Open':
        return '#4CAF50';
      case 'Closed':
        return '#F44336';
      default:
        return '#FF9800';
    }
  };

  const getChangeColor = (change) => {
    return change >= 0 ? '#4CAF50' : '#F44336';
  };

  return (
    <Card
      sx={{
        background: 'rgba(21, 25, 50, 0.8)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <CardContent sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 3 }}>
          Market Overview
        </Typography>

        {/* Market Status */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Market Status
            </Typography>
            <Chip
              label={marketStatus.status}
              size="small"
              sx={{
                backgroundColor: getStatusColor(marketStatus.status),
                color: 'white',
                fontWeight: 'bold',
              }}
            />
          </Box>

          {marketStatus.nextOpen && (
            <Typography variant="body2" color="text.secondary">
              Next opening: {marketStatus.nextOpen} ({marketStatus.timeRemaining})
            </Typography>
          )}

          {/* Market Hours Progress */}
          <LinearProgress
            variant="determinate"
            value={isConnected ? 75 : 0}
            sx={{
              height: 6,
              borderRadius: 3,
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: getStatusColor(marketStatus.status),
                borderRadius: 3,
              },
            }}
          />
        </Box>

        {/* Key Indices */}
        <Typography variant="subtitle2" sx={{ mb: 2 }}>
          Key Indices
        </Typography>

        <List sx={{ p: 0 }}>
          {indices.map((index, idx) => (
            <ListItem
              key={idx}
              sx={{
                px: 0,
                py: 1,
                borderBottom: idx < indices.length - 1 ? '1px solid rgba(255, 255, 255, 0.1)' : 'none',
              }}
            >
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {index.name}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                      {index.value.toLocaleString('en-IN')}
                    </Typography>
                  </Box>
                }
                secondary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {index.change >= 0 ? (
                      <TrendingUp sx={{ fontSize: 16, color: getChangeColor(index.change) }} />
                    ) : (
                      <TrendingDown sx={{ fontSize: 16, color: getChangeColor(index.change) }} />
                    )}
                    <Typography
                      variant="caption"
                      sx={{
                        color: getChangeColor(index.change),
                        fontWeight: 'bold',
                      }}
                    >
                      {index.change >= 0 ? '+' : ''}
                      {index.change.toFixed(2)} ({index.changePercent >= 0 ? '+' : ''}
                      {index.changePercent}%)
                    </Typography>
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>

        {/* Connection Status */}
        <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {isConnected ? (
              <Speed sx={{ fontSize: 16, color: '#4CAF50' }} />
            ) : (
              <NotificationsActive sx={{ fontSize: 16, color: '#FF9800' }} />
            )}
            <Typography variant="body2" color="text.secondary">
              {isConnected ? 'Real-time data connected' : 'Real-time data disconnected'}
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default MarketOverview;