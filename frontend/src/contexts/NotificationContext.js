/**
 * Notification Context
 * Manages real-time notifications and alerts for the trading dashboard
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const NotificationContext = createContext();

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);

  // Show toast notification
  const showToast = useCallback((message, type = 'info', options = {}) => {
    toast[type](message, {
      position: 'top-right',
      autoClose: 5000,
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      ...options,
    });
  }, []);

  // Add notification
  const addNotification = useCallback((notification) => {
    const newNotification = {
      id: Date.now(),
      timestamp: new Date(),
      read: false,
      ...notification,
    };

    setNotifications(prev => [newNotification, ...prev]);
    setUnreadCount(prev => prev + 1);

    // Show toast for important notifications
    if (notification.type === 'error' || notification.type === 'success') {
      showToast(notification.message, notification.type === 'error' ? 'error' : 'success');
    }

    return newNotification.id;
  }, []);

  // Mark notification as read
  const markAsRead = useCallback((id) => {
    setNotifications(prev =>
      prev.map(notification =>
        notification.id === id ? { ...notification, read: true } : notification
      )
    );
    setUnreadCount(prev => Math.max(0, prev - 1));
  }, []);

  // Mark all notifications as read
  const markAllAsRead = useCallback(() => {
    setNotifications(prev =>
      prev.map(notification => ({ ...notification, read: true }))
    );
    setUnreadCount(0);
  }, []);

  // Remove notification
  const removeNotification = useCallback((id) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
    }, []);

  // Clear all notifications
  const clearNotifications = useCallback(() => {
    setNotifications([]);
    setUnreadCount(0);
  }, []);

  // Success notification helper
  const showSuccess = useCallback((message, options = {}) => {
    const id = addNotification({
      type: 'success',
      message,
      title: 'Success',
      ...options,
    });
    return id;
  }, [addNotification]);

  // Error notification helper
  const showError = useCallback((message, options = {}) => {
    const id = addNotification({
      type: 'error',
      message,
      title: 'Error',
      ...options,
    });
    return id;
  }, [addNotification]);

  // Warning notification helper
  const showWarning = useCallback((message, options = {}) => {
    const id = addNotification({
      type: 'warning',
      message,
      title: 'Warning',
      ...options,
    });
    return id;
  }, [addNotification]);

  // Info notification helper
  const showInfo = useCallback((message, options = {}) => {
    const id = addNotification({
      type: 'info',
      message,
      title: 'Information',
      ...options,
    });
    return id;
  }, [addNotification]);

  // Signal notification helper
  const showSignal = useCallback((signal, options = {}) => {
    const { signal_type, symbol, confidence_score, potential_profit_pct } = signal;

    let message = `${signal_type} signal for ${symbol}`;
    let type = 'info';

    if (confidence_score >= 9) {
      type = 'success';
      message += ` (High Confidence: ${confidence_score}/10)`;
    }

    if (potential_profit_pct && potential_profit_pct > 1) {
      message += ` - Potential: ${potential_profit_pct}%`;
    }

    const id = addNotification({
      type: 'signal',
      message,
      title: `${signal_type} Signal`,
      signal,
      ...options,
    });

    // Play sound for high-confidence signals
    if (confidence_score >= 9) {
      // Optionally play a sound
      // playSound('signal-high');
    }

    return id;
  }, [addNotification]);

  // Connection notification helper
  const showConnectionStatus = useCallback((status, options = {}) => {
    let message = '';
    let type = 'info';

    switch (status) {
      case 'connected':
        message = 'Connected to real-time data stream';
        type = 'success';
        break;
      case 'disconnected':
        message = 'Disconnected from real-time data stream';
        type = 'warning';
        break;
      case 'reconnecting':
        message = 'Reconnecting to real-time data stream...';
        type = 'info';
        break;
      case 'error':
        message = 'Connection error - attempting to reconnect';
        type = 'error';
        break;
      default:
        message = `Connection status: ${status}`;
    }

    return addNotification({
      type: 'connection',
      message,
      title: 'Connection Status',
      ...options,
    });
  }, [addNotification]);

  // Handle WebSocket signal events
  useEffect(() => {
    const handleNewSignal = (event) => {
      const signal = event.detail;
      showSignal(signal);
    };

    window.addEventListener('newSignal', handleNewSignal);
    return () => window.removeEventListener('newSignal', handleNewSignal);
  }, [showSignal]);

  // Auto-remove old notifications (older than 24 hours)
  useEffect(() => {
    const interval = setInterval(() => {
      const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
      setNotifications(prev =>
        prev.filter(notification => notification.timestamp > twentyFourHoursAgo)
      );
    }, 60 * 60 * 1000); // Check every hour

    return () => clearInterval(interval);
  }, []);

  const value = {
    notifications,
    unreadCount,
    addNotification,
    markAsRead,
    markAllAsRead,
    removeNotification,
    clearNotifications,
    showToast,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showSignal,
    showConnectionStatus,
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};