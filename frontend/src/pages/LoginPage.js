/**
 * Professional Login Page
 * Handles Zerodha Kite Connect authentication
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Alert,
  Stepper,
  Step,
  StepLabel,
  Paper,
  Divider,
  Link,
  CircularProgress,
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';

const LoginPage = () => {
  const { login, isLoading } = useAuth();
  const [step, setStep] = useState(0);
  const [credentials, setCredentials] = useState({
    api_key: '',
    api_secret: '',
    request_token: '',
  });
  const [loginUrl, setLoginUrl] = useState('');
  const [error, setError] = useState('');

  const handleGenerateLoginUrl = () => {
    if (!credentials.api_key || !credentials.api_secret) {
      setError('Please enter both API Key and API Secret');
      return;
    }

    const url = `https://kite.zerodha.com/connect/login?api_key=${credentials.api_key}&v=3`;
    setLoginUrl(url);
    setStep(1);
    setError('');
  };

  const handleLogin = async () => {
    if (!credentials.request_token) {
      setError('Please enter the request token from the redirect URL');
      return;
    }

    const success = await login(credentials);
    if (!success) {
      setError('Login failed. Please check your credentials and try again.');
    }
  };

  const handleInputChange = (field) => (event) => {
    setCredentials(prev => ({
      ...prev,
      [field]: event.target.value,
    }));
    };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        p: 3,
      }}
    >
      <Card
        sx={{
          maxWidth: 500,
          width: '100%',
          background: 'rgba(255, 255, 255, 0.95)',
          backdropFilter: 'blur(10px)',
        }}
      >
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom align="center" sx={{ fontWeight: 'bold' }}>
            Professional Trading Dashboard
          </Typography>

          <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 4 }}>
            Login with your Zerodha Kite Connect credentials
          </Typography>

          <Stepper activeStep={step} sx={{ mb: 4 }}>
            <Step>
              <StepLabel>Credentials</StepLabel>
            </Step>
            <Step>
              <StepLabel>Authorization</StepLabel>
            </Step>
            <Step>
              <StepLabel>Complete</StepLabel>
            </Step>
          </Stepper>

          {error && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          {step === 0 && (
            <Box>
              <TextField
                fullWidth
                label="API Key"
                value={credentials.api_key}
                onChange={handleInputChange('api_key')}
                margin="normal"
                variant="outlined"
                helperText="Your Zerodha Kite Connect API Key"
              />
              <TextField
                fullWidth
                label="API Secret"
                type="password"
                value={credentials.api_secret}
                onChange={handleInputChange('api_secret')}
                margin="normal"
                variant="outlined"
                helperText="Your Zerodha Kite Connect API Secret"
              />

              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={handleGenerateLoginUrl}
                sx={{ mt: 3 }}
              >
                Generate Login URL
              </Button>
            </Box>
          )}

          {step === 1 && (
            <Box>
              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2" gutterBottom>
                  <strong>Step 2: Authorization</strong>
                </Typography>
                <Typography variant="body2">
                  1. Click the button below to open the Zerodha login page
                  <br />
                  2. Login with your Zerodha credentials
                  <br />
                  3. Authorize the application
                  <br />
                  4. Copy the <code>request_token</code> from the redirect URL
                </Typography>
              </Alert>

              {loginUrl && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="body2" gutterBottom>
                    Login URL:
                  </Typography>
                  <Paper sx={{ p: 2, backgroundColor: '#f5f5f5', wordBreak: 'break-all' }}>
                    <Link href={loginUrl} target="_blank" rel="noopener noreferrer">
                      {loginUrl}
                    </Link>
                  </Paper>
                </Box>
              )}

              <TextField
                fullWidth
                label="Request Token"
                value={credentials.request_token}
                onChange={handleInputChange('request_token')}
                margin="normal"
                variant="outlined"
                helperText="Paste the request_token from the redirect URL"
              />

              <Button
                fullWidth
                variant="contained"
                size="large"
                onClick={handleLogin}
                disabled={isLoading}
                sx={{ mt: 3 }}
              >
                {isLoading ? (
                  <>
                    <CircularProgress size={20} sx={{ mr: 1 }} />
                    Logging in...
                  </>
                ) : (
                  'Complete Login'
                )}
              </Button>

              <Button
                fullWidth
                variant="text"
                onClick={() => setStep(0)}
                sx={{ mt: 2 }}
              >
                Back
              </Button>
            </Box>
          )}

          {step === 2 && (
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>
                Authentication Successful!
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Redirecting to dashboard...
              </Typography>
            </Box>
          )}

          <Divider sx={{ my: 3 }} />

          <Typography variant="caption" color="text.secondary" align="center">
            Don't have a Kite Connect account?{' '}
            <Link href="https://kite.trade/" target="_blank" rel="noopener noreferrer">
              Sign up for free
            </Link>
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default LoginPage;