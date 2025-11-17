/**
 * WebSocket Context
 * Manages real-time WebSocket connections for live trading data
 */

import React, { createContext, useContext, useEffect, useState, useRef } from 'react';

const WebSocketContext = createContext();

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [lastMessage, setLastMessage] = useState(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastPing, setLastPing] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const pingIntervalRef = useRef(null);

  const connect = () => {
    const token = localStorage.getItem('authToken');
    if (!token) {
      console.warn('No auth token found for WebSocket connection');
      return;
    }

    try {
      // Close existing connection if any
      if (wsRef.current) {
        wsRef.current.close();
      }

      // Create new WebSocket connection
      const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws?token=${token}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setConnectionStatus('connected');
        setReconnectAttempts(0);

        // Start ping interval
        startPingInterval();
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);

          // Handle different message types
          switch (data.type) {
            case 'pong':
              setLastPing(Date.now());
              break;
            case 'new_signal':
              // Handle new trading signal
              handleNewSignal(data.signal);
              break;
            case 'connection_established':
              console.log('WebSocket connection established for user:', data.user_id);
              break;
            case 'recent_signals':
              // Handle recent signals
              handleRecentSignals(data.signals);
              break;
            default:
              console.log('Received WebSocket message:', data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        setIsConnected(false);
        setConnectionStatus('disconnected');
        stopPingInterval();

        // Attempt reconnection if not a normal close
        if (event.code !== 1000) {
          scheduleReconnect();
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setConnectionStatus('error');
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setConnectionStatus('error');
    }
  };

  const disconnect = () => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    stopPingInterval();

    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }

    setIsConnected(false);
    setConnectionStatus('disconnected');
  };

  const scheduleReconnect = () => {
    const maxReconnectAttempts = 5;
    const baseDelay = 1000; // 1 second

    if (reconnectAttempts >= maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      setConnectionStatus('failed');
      return;
    }

    const delay = baseDelay * Math.pow(2, reconnectAttempts);
    setReconnectAttempts(prev => prev + 1);
    setConnectionStatus('reconnecting');

    reconnectTimeoutRef.current = setTimeout(() => {
      console.log(`Attempting to reconnect (${reconnectAttempts + 1}/${maxReconnectAttempts})`);
      connect();
    }, delay);
  };

  const startPingInterval = () => {
    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000); // Ping every 30 seconds
  };

  const stopPingInterval = () => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  };

  const sendMessage = (message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  };

  const handleNewSignal = (signal) => {
    // Dispatch custom event for signal notifications
    window.dispatchEvent(new CustomEvent('newSignal', { detail: signal }));
  };

  const handleRecentSignals = (signals) => {
    // Dispatch custom event for recent signals
    window.dispatchEvent(new CustomEvent('recentSignals', { detail: signals }));
  };

  useEffect(() => {
    // Connect when component mounts and user is authenticated
    const token = localStorage.getItem('authToken');
    if (token) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, []);

  const value = {
    isConnected,
    connectionStatus,
    lastMessage,
    reconnectAttempts,
    lastPing,
    connect,
    disconnect,
    sendMessage,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};