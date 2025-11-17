/**
 * Professional Trading Dashboard - Main Application
 * React frontend with Material-UI and real-time WebSocket integration
 */

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';

import AuthProvider from './contexts/AuthContext';
import WebSocketProvider from './contexts/WebSocketContext';
import NotificationProvider from './contexts/NotificationContext';

import Layout from './components/Layout/Layout';
import LoginPage from './pages/LoginPage';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import LoadingScreen from './components/UI/LoadingScreen';
import ErrorBoundary from './components/ErrorBoundary';

// Create a professional theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00E676', // Professional green
      light: '#00C853',
      dark: '#00B248',
      contrastText: '#000000',
    },
    secondary: {
      main: '#FF6B6B', // Professional red for sell signals
      light: '#FF8787',
      dark: '#FF5252',
      contrastText: '#000000',
    },
    background: {
      default: '#0A0E27', // Dark professional background
      paper: '#151932',
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#B0BEC5',
    },
    success: {
      main: '#00E676',
    },
    warning: {
      main: '#FFD600',
    },
    error: {
      main: '#FF5252',
    },
    info: {
      main: '#448AFF',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.5,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(21, 25, 50, 0.8)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: 16,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
  },
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 2,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check for existing authentication
    const token = localStorage.getItem('authToken');
    if (token) {
      // Validate token with backend
      validateToken(token);
    } else {
      setIsLoading(false);
    }
  }, []);

  const validateToken = async (token) => {
    try {
      const response = await fetch('/api/auth/validate', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        setIsAuthenticated(true);
      } else {
        localStorage.removeItem('authToken');
      }
    } catch (error) {
      console.error('Token validation failed:', error);
      localStorage.removeItem('authToken');
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <AuthProvider>
            <WebSocketProvider>
              <NotificationProvider>
                <Router>
                  <Routes>
                    <Route
                      path="/login"
                      element={
                        isAuthenticated ? (
                          <Navigate to="/dashboard" replace />
                        ) : (
                          <LoginPage />
                        )
                      }
                    />
                    <Route
                      path="/"
                      element={
                        isAuthenticated ? (
                          <Layout />
                        ) : (
                          <Navigate to="/login" replace />
                        )
                      }
                    >
                      <Route index element={<Navigate to="/dashboard" replace />} />
                      <Route path="dashboard" element={<Dashboard />} />
                      <Route path="analytics" element={<Analytics />} />
                      <Route path="settings" element={<Settings />} />
                    </Route>
                    <Route
                      path="*"
                      element={
                        isAuthenticated ? (
                          <Navigate to="/dashboard" replace />
                        ) : (
                          <Navigate to="/login" replace />
                        )
                      }
                    />
                  </Routes>
                </Router>
              </NotificationProvider>
            </WebSocketProvider>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;