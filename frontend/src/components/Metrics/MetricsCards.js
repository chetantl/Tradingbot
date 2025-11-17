/**
 * Metrics Cards Component
 * Displays key trading metrics and system statistics
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp,
  Timeline,
  Assessment,
  Speed,
  NotificationsActive,
  People,
} from '@mui/icons-material';

const MetricsCards = ({ marketData }) => {
  // Mock data for demonstration
  const metrics = [
    {
      title: 'Active Signals',
      value: marketData.total_signals || 0,
      icon: <TrendingUp />,
      color: '#4CAF50',
      bgColor: 'rgba(76, 175, 80, 0.1)',
      change: '+12%',
      positive: true,
    },
    {
      title: 'Monitored Symbols',
      value: marketData.monitored_symbols || 0,
      icon: <Speed />,
      color: '#2196F3',
      bgColor: 'rgba(33, 150, 243, 0.1)',
      change: '+2',
      positive: true,
    },
    {
      title: 'Success Rate',
      value: '78%',
      icon: <Assessment />,
      color: '#FF9800',
      bgColor: 'rgba(255, 152, 0, 0.1)',
      change: '+5%',
      positive: true,
    },
    {
      title: 'Active Users',
      value: marketData.active_users || 1,
      icon: <People />,
      color: '#9C27B0',
      bgColor: 'rgba(156, 39, 176, 0.1)',
      change: 'You',
      positive: true,
    },
  ];

  const MetricCard = ({ metric }) => (
    <Card
      sx={{
        height: '100%',
        background: 'rgba(21, 25, 50, 0.8)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        },
      }}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              p: 1.5,
              borderRadius: 2,
              backgroundColor: metric.bgColor,
              color: metric.color,
            }}
          >
            {metric.icon}
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="body2" sx={{ color: metric.positive ? '#4CAF50' : '#F44336' }}>
              {metric.change}
            </Typography>
          </Box>
        </Box>

        <Typography variant="h4" component="div" sx={{ fontWeight: 'bold', mb: 1 }}>
          {metric.value}
        </Typography>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {metric.title}
        </Typography>

        {/* Progress Bar */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ flex: 1 }}>
            <LinearProgress
              variant="determinate"
              value={75}
              sx={{
                height: 4,
                borderRadius: 2,
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: metric.color,
                  borderRadius: 2,
                },
              }}
            />
          </Box>
          <Typography variant="caption" color="text.secondary">
            75%
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Grid container spacing={3}>
      {metrics.map((metric, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <MetricCard metric={metric} />
        </Grid>
      ))}
    </Grid>
  );
};

export default MetricsCards;