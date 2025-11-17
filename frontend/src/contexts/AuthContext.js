/**
 * Authentication Context
 * Manages user authentication state and provides auth-related functions
 */

import React, { createContext, useContext, useState, useEffect } from 'react';
import { message } from 'antd';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Check for existing session on mount
    const token = localStorage.getItem('authToken');
    const userData = localStorage.getItem('userData');

    if (token && userData) {
      setUser(JSON.parse(userData));
      setIsAuthenticated(true);
    }
  }, []);

  const login = async (credentials) => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Login failed');
      }

      // Store authentication data
      localStorage.setItem('authToken', data.access_token);
      localStorage.setItem('userData', JSON.stringify(data.user_info));

      setUser(data.user_info);
      setIsAuthenticated(true);

      message.success('Login successful!');
      return true;

    } catch (error) {
      console.error('Login error:', error);
      message.error(error.message || 'Login failed');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    setIsLoading(true);
    try {
      const token = localStorage.getItem('authToken');

      if (token) {
        // Notify backend about logout
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
        });
      }

      // Clear local storage
      localStorage.removeItem('authToken');
      localStorage.removeItem('userData');

      setUser(null);
      setIsAuthenticated(false);

      message.success('Logged out successfully');

    } catch (error) {
      console.error('Logout error:', error);
      // Still logout locally even if backend call fails
      localStorage.removeItem('authToken');
      localStorage.removeItem('userData');
      setUser(null);
      setIsAuthenticated(false);
    } finally {
      setIsLoading(false);
    }
  };

  const refreshToken = async () => {
    try {
      const token = localStorage.getItem('authToken');
      if (!token) return false;

      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('authToken', data.access_token);
        return true;
      } else {
        // Token refresh failed, logout user
        await logout();
        return false;
      }
    } catch (error) {
      console.error('Token refresh error:', error);
      await logout();
      return false;
    }
  };

  const value = {
    user,
    isAuthenticated,
    isLoading,
    login,
    logout,
    refreshToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};