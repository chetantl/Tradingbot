/**
 * Professional Signal Card Component
 * Displays individual trading signals with comprehensive information
 */

import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  Speed,
  Assessment,
  Info,
  Star,
} from '@mui/icons-material';

const SignalCard = ({ signal, index }) => {
  // Determine signal color and icon
  const getSignalConfig = (signalType) => {
    const configs = {
      ACCUMULATION: {
        color: '#4CAF50',
        bgColor: 'rgba(76, 175, 80, 0.1)',
        icon: <TrendingUp />,
        label: 'ACCUMULATION',
        description: 'Hidden buying detected'
      },
      DISTRIBUTION: {
        color: '#F44336',
        bgColor: 'rgba(244, 67, 54, 0.1)',
        icon: <TrendingDown />,
        label: 'DISTRIBUTION',
        description: 'Hidden selling detected'
      },
      BUY: {
        color: '#2196F3',
        bgColor: 'rgba(33, 150, 243, 0.1)',
        icon: <TrendingUp />,
        label: 'BUY',
        description: 'Strong buying pressure'
      },
      SELL: {
        color: '#FF9800',
        bgColor: 'rgba(255, 152, 0, 0.1)',
        icon: <TrendingDown />,
        label: 'SELL',
        description: 'Strong selling pressure'
      }
    };
    return configs[signalType] || configs.BUY;
  };

  const config = getSignalConfig(signal.signal_type);
  const confidence = signal.confidence_score || 0;
  const profit = signal.potential_profit_pct || 0;

  // Format price
  const formatPrice = (price) => {
    return `₹${price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  // Get confidence color
  const getConfidenceColor = (score) => {
    if (score >= 9) return '#4CAF50'; // Green
    if (score >= 7) return '#FF9800'; // Orange
    return '#F44336'; // Red
  };

  // Get PCR bias color
  const getPCRColor = (bias) => {
    if (bias?.includes('BULLISH')) return '#4CAF50';
    if (bias?.includes('BEARISH')) return '#F44336';
    return '#9E9E9E'; // Neutral
  };

  return (
    <Card
      sx={{
        mb: 2,
        borderLeft: `4px solid ${config.color}`,
        background: 'rgba(21, 25, 50, 0.8)',
        backdropFilter: 'blur(10px)',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
        },
      }}
    >
      <CardContent sx={{ p: 2 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ color: config.color, p: 1, borderRadius: 1, backgroundColor: config.bgColor }}>
              {config.icon}
            </Box>
            <Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: config.color }}>
                {signal.symbol}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {config.label} • {config.description}
              </Typography>
            </Box>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="Confidence Score">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Star sx={{ fontSize: 16, color: getConfidenceColor(confidence) }} />
                <Typography variant="body2" sx={{ fontWeight: 'bold', color: getConfidenceColor(confidence) }}>
                  {confidence}/10
                </Typography>
              </Box>
            </Tooltip>

            {profit > 0 && (
              <Chip
                label={`+${profit}%`}
                size="small"
                color={profit > 2 ? 'success' : 'primary'}
                sx={{ fontSize: '0.75rem' }}
              />
            )}
          </Box>
        </Box>

        {/* Price Information */}
        <Grid container spacing={2} sx={{ mb: 2 }}>
          <Grid item xs={3}>
            <Typography variant="caption" color="text.secondary">
              Current
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
              {formatPrice(signal.current_price)}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="caption" color="text.secondary">
              Entry
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
              {formatPrice(signal.entry_price)}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="caption" color="text.secondary">
              Target
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#4CAF50' }}>
              {formatPrice(signal.target_price)}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="caption" color="text.secondary">
              Stop Loss
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#F44336' }}>
              {formatPrice(signal.stop_loss)}
            </Typography>
          </Grid>
        </Grid>

        {/* Key Metrics */}
        <Box sx={{ mb: 2 }}>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Speed sx={{ fontSize: 16, color: '#9E9E9E' }} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Institutional Ratio
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                    {signal.institutional_ratio}
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Assessment sx={{ fontSize: 16, color: '#9E9E9E' }} />
                <Box sx={{ flex: 1 }}>
                  <Typography variant="caption" color="text.secondary">
                    Volume Status
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                    {signal.volume_status}
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Box>

        {/* Progress Bars */}
        <Box sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
            <Typography variant="caption" color="text.secondary">
              Confidence
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {confidence}/10
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={confidence * 10}
            sx={{
              height: 6,
              borderRadius: 3,
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              '& .MuiLinearProgress-bar': {
                backgroundColor: getConfidenceColor(confidence),
                borderRadius: 3,
              },
            }}
          />
        </Box>

        {/* Additional Info */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Chip
              label={`R:R ${signal.risk_reward}`}
              size="small"
              variant="outlined"
              sx={{ fontSize: '0.7rem', height: 24 }}
            />
            <Chip
              label={signal.pcr_bias}
              size="small"
              variant="outlined"
              sx={{
                fontSize: '0.7rem',
                height: 24,
                color: getPCRColor(signal.pcr_bias),
                borderColor: getPCRColor(signal.pcr_bias),
              }}
            />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Timeline sx={{ fontSize: 14, color: 'text.secondary' }} />
            <Typography variant="caption" color="text.secondary">
              {signal.time_detected}
            </Typography>
            <Tooltip title="View Details">
              <IconButton size="small" sx={{ p: 0.5 }}>
                <Info sx={{ fontSize: 14, color: 'text.secondary' }} />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SignalCard;