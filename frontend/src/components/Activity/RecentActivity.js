/**
 * Recent Activity Component
 * Shows recent trading signals and system events
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  NotificationsActive,
  Assessment,
} from '@mui/icons-material';

const RecentActivity = () => {
  const [activities, setActivities] = useState([]);

  useEffect(() => {
    // Mock recent activities
    const mockActivities = [
      {
        id: 1,
        type: 'signal',
        signal_type: 'ACCUMULATION',
        symbol: 'RELIANCE',
        confidence: 9,
        message: 'High confidence accumulation signal detected',
        timestamp: new Date(Date.now() - 5 * 60 * 1000),
      },
      {
        id: 2,
        type: 'signal',
        signal_type: 'BUY',
        symbol: 'TCS',
        confidence: 8,
        message: 'Strong buying pressure identified',
        timestamp: new Date(Date.now() - 15 * 60 * 1000),
      },
      {
        id: 3,
        type: 'system',
        message: 'WebSocket connection established',
        timestamp: new Date(Date.now() - 30 * 60 * 1000),
      },
      {
        id: 4,
        type: 'signal',
        signal_type: 'DISTRIBUTION',
        symbol: 'INFY',
        confidence: 7,
        message: 'Distribution pattern detected',
        timestamp: new Date(Date.now() - 45 * 60 * 1000),
      },
    ];

    setActivities(mockActivities);
  }, []);

  const getActivityIcon = (activity) => {
    if (activity.type === 'signal') {
      switch (activity.signal_type) {
        case 'ACCUMULATION':
        case 'BUY':
          return <TrendingUp color="#4CAF50" />;
        case 'DISTRIBUTION':
        case 'SELL':
          return <TrendingDown color="#F44336" />;
        default:
          return <Assessment color="#2196F3" />;
      }
    }

    if (activity.type === 'system') {
      return <NotificationsActive color="#FF9800" />;
    }

    return <Timeline color="#9E9E9E" />;
  };

  const formatTime = (timestamp) => {
    const now = new Date();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ago`;
    } else if (minutes > 0) {
      return `${minutes}m ago`;
    } else {
      return 'Just now';
    }
  };

  const getSignalColor = (signalType) => {
    const colors = {
      ACCUMULATION: '#4CAF50',
      DISTRIBUTION: '#F44336',
      BUY: '#2196F3',
      SELL: '#FF9800',
    };
    return colors[signalType] || '#9E9E9E';
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
        <Typography variant="h6" sx={{ mb: 2 }}>
          Recent Activity
        </Typography>

        <List sx={{ p: 0 }}>
          {activities.map((activity, index) => (
            <ListItem
              key={activity.id}
              sx={{
                px: 0,
                py: 1,
                borderBottom: index < activities.length - 1 ? '1px solid rgba(255, 255, 255, 0.1)' : 'none',
              }}
            >
              <ListItemIcon sx={{ minWidth: 40 }}>
                {getActivityIcon(activity)}
              </ListItemIcon>
              <ListItemText
                primary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {activity.message}
                    </Typography>
                    {activity.type === 'signal' && (
                      <Chip
                        label={activity.signal_type}
                        size="small"
                        sx={{
                          fontSize: '0.7rem',
                          height: 20,
                          backgroundColor: getSignalColor(activity.signal_type),
                          color: 'white',
                        }}
                      />
                    )}
                  </Box>
                }
                secondary={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      {activity.symbol && `${activity.symbol} • `}
                      {formatTime(activity.timestamp)}
                    </Typography>
                    {activity.type === 'signal' && (
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Typography variant="caption" color="text.secondary">
                          {activity.confidence}/10
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={activity.confidence * 10}
                          sx={{
                            width: 40,
                            height: 3,
                            borderRadius: 1.5,
                            backgroundColor: 'rgba(255, 255, 255, 0.1)',
                            '& .MuiLinearProgress-bar': {
                              backgroundColor: getSignalColor(activity.signal_type),
                              borderRadius: 1.5,
                            },
                          }}
                        />
                      </Box>
                    )}
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>

        {activities.length === 0 && (
          <Box
            sx={{
              textAlign: 'center',
              py: 4,
              color: 'text.secondary',
            }}
          >
            <Timeline sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
            <Typography variant="body2">
              No recent activity
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default RecentActivity;