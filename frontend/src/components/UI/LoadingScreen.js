/**
 * Loading Screen Component
 * Professional loading animation with branding
 */

import React from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { TrendingUp } from '@mui/icons-material';

const LoadingScreen = () => {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
      }}
    >
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <TrendingUp sx={{ fontSize: 64, mb: 2, animation: 'pulse 2s infinite' }} />
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          Professional Trading Dashboard
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.8 }}>
          Institutional Order Flow Detection System
        </Typography>
      </Box>

      <CircularProgress size={40} sx={{ color: 'white' }} />
      <Typography variant="body1" sx={{ mt: 2, opacity: 0.8 }}>
        Loading...
      </Typography>
    </Box>
  );
};

export default LoadingScreen;