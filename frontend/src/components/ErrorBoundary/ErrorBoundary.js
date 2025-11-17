/**
 * Error Boundary Component
 * Catches JavaScript errors and displays fallback UI
 */

import React from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Alert,
} from '@mui/material';
import { Refresh } from '@mui/icons-material';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error, errorInfo) {
    return {
      hasError: true,
      error,
      errorInfo,
    };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            p: 3,
            background: 'linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%)',
          }}
        >
          <Paper sx={{ p: 4, maxWidth: 600, textAlign: 'center' }}>
            <Alert severity="error" sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Application Error
              </Typography>
              <Typography variant="body2">
                Something went wrong while loading the application.
              </Typography>
            </Alert>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Error details: {this.state.error?.message || 'Unknown error occurred'}
            </Typography>

            {process.env.NODE_ENV === 'development' && (
              <Box sx={{ mt: 2, p: 2, backgroundColor: 'rgba(0, 0, 0, 0.1)', borderRadius: 1 }}>
                <Typography variant="body2" component="pre" sx={{ fontSize: '0.8rem', textAlign: 'left' }}>
                  {this.state.error?.stack}
                </Typography>
              </Box>
            )}

            <Button
              variant="contained"
              startIcon={<Refresh />}
              onClick={this.handleRetry}
              sx={{ mt: 2 }}
            >
              Try Again
            </Button>
          </Paper>
        </Box>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;