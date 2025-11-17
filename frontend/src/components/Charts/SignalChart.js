/**
 * Signal Chart Component
 * Displays signal distribution and performance over time
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

const SignalChart = ({ signals }) => {
  // Prepare data for charts
  const getSignalTypeData = () => {
    const typeCounts = signals.reduce((acc, signal) => {
      acc[signal.signal_type] = (acc[signal.signal_type] || 0) + 1;
      return acc;
    }, {});

    return Object.entries(typeCounts).map(([name, value]) => ({
      name,
      value,
      color: getSignalColor(name),
    }));
  };

  const getSignalTimeData = () => {
    return signals.slice(0, 20).map((signal, index) => ({
      time: signal.time_detected,
      confidence: signal.confidence_score,
      profit: signal.potential_profit_pct || 0,
    }));
  };

  const getSignalColor = (type) => {
    const colors = {
      ACCUMULATION: '#4CAF50',
      DISTRIBUTION: '#F44336',
      BUY: '#2196F3',
      SELL: '#FF9800',
    };
    return colors[type] || '#9E9E9E';
  };

  const signalTypeData = getSignalTypeData();
  const signalTimeData = getSignalTimeData();

  return (
    <Card
      sx={{
        height: 400,
        background: 'rgba(21, 25, 50, 0.8)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
      }}
    >
      <CardContent sx={{ p: 3, height: '100%' }}>
        <Typography variant="h6" sx={{ mb: 3 }}>
          Signal Analysis
        </Typography>

        {signalTypeData.length > 0 ? (
          <Box sx={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={signalTypeData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {signalTypeData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(value, name) => [value, name]}
                  contentStyle={{
                    backgroundColor: 'rgba(21, 25, 50, 0.9)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: '8px',
                  }}
                />
                <Legend
                  verticalAlign="middle"
                  align="right"
                  layout="vertical"
                  formatter={(value, entry) => `${entry.payload.name}: ${value}`}
                />
              </PieChart>
            </ResponsiveContainer>
          </Box>
        ) : (
          <Box
            sx={{
              height: 300,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'text.secondary',
            }}
          >
            <Typography variant="body2">
              No signal data available for charting
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default SignalChart;