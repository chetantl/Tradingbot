/**
 * Settings Page
 * Configuration and preferences management
 */

import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemSecondary,
  Chip,
  Alert,
} from '@mui/material';
import { useAuth } from '../contexts/AuthContext';

const Settings = () => {
  const { user, logout } = useAuth();
  const [settings, setSettings] = useState({
    notifications: true,
    soundAlerts: true,
    darkMode: true,
    autoRefresh: true,
    refreshInterval: 60,
    confidenceThreshold: 7,
    maxSignals: 10,
  });

  const [successMessage, setSuccessMessage] = useState('');

  const handleSettingChange = (key) => (event) => {
    const value = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    setSettings(prev => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleSaveSettings = () => {
    // Save settings to localStorage or backend
    localStorage.setItem('dashboard_settings', JSON.stringify(settings));
    setSuccessMessage('Settings saved successfully!');
    setTimeout(() => setSuccessMessage(''), 3000);
  };

  const handleLogout = () => {
    logout();
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      {successMessage && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccessMessage('')}>
          {successMessage}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Account Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Account Settings
              </Typography>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  <strong>API Key:</strong> {user?.api_key || 'N/A'}
                </Typography>
                <Typography variant="body2" gutterBottom>
                  <strong>User:</strong> {user?.user_name || 'N/A'}
                </Typography>
                <Typography variant="body2">
                  <strong>Status:</strong> <Chip label="Connected" color="success" size="small" />
                </Typography>
              </Box>

              <Button
                variant="outlined"
                color="error"
                onClick={handleLogout}
                sx={{ mt: 2 }}
              >
                Logout
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Notification Settings
              </Typography>

              <List>
                <ListItem>
                  <ListItemText
                    primary="Real-time Notifications"
                    secondary="Get notified when new signals are generated"
                  />
                  <ListItemSecondary>
                    <Switch
                      checked={settings.notifications}
                      onChange={handleSettingChange('notifications')}
                    />
                  </ListItemSecondary>
                </ListItem>

                <ListItem>
                  <ListItemText
                    primary="Sound Alerts"
                    secondary="Play sound for high-confidence signals"
                  />
                  <ListItemSecondary>
                    <Switch
                      checked={settings.soundAlerts}
                      onChange={handleSettingChange('soundAlerts')}
                    />
                  </ListItemSecondary>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Display Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Display Settings
              </Typography>

              <List>
                <ListItem>
                  <ListItemText
                    primary="Dark Mode"
                    secondary="Use dark theme for the dashboard"
                  />
                  <ListItemSecondary>
                    <Switch
                      checked={settings.darkMode}
                      onChange={handleSettingChange('darkMode')}
                    />
                  </ListItemSecondary>
                </ListItem>

                <ListItem>
                  <ListItemText
                    primary="Auto Refresh"
                    secondary="Automatically refresh data at regular intervals"
                  />
                  <ListItemSecondary>
                    <Switch
                      checked={settings.autoRefresh}
                      onChange={handleSettingChange('autoRefresh')}
                    />
                  </ListItemSecondary>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Trading Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trading Settings
              </Typography>

              <TextField
                fullWidth
                label="Refresh Interval (seconds)"
                type="number"
                value={settings.refreshInterval}
                onChange={handleSettingChange('refreshInterval')}
                margin="normal"
                variant="outlined"
                helperText="How often to refresh data automatically"
              />

              <TextField
                fullWidth
                label="Confidence Threshold"
                type="number"
                value={settings.confidenceThreshold}
                onChange={handleSettingChange('confidenceThreshold')}
                margin="normal"
                variant="outlined"
                helperText="Minimum confidence score for signals (1-10)"
                inputProps={{ min: 1, max: 10 }}
              />

              <TextField
                fullWidth
                label="Maximum Signals"
                type="number"
                value={settings.maxSignals}
                onChange={handleSettingChange('maxSignals')}
                margin="normal"
                variant="outlined"
                helperText="Maximum number of signals to display"
                inputProps={{ min: 1, max: 50 }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Save Button */}
        <Grid item xs={12}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Button
                variant="contained"
                size="large"
                onClick={handleSaveSettings}
                sx={{ minWidth: 200 }}
              >
                Save Settings
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;