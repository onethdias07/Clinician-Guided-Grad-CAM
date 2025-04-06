import React, { useState } from 'react';
import axios from 'axios';

const LoginForm = ({ onLoginSuccess, onSwitchToRegister }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [loginError, setLoginError] = useState('');
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  
  const handleInputChange = (e) => {
    const { id, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [id]: value
    }));
  };
  
  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    const { username, password } = formData;
    
    setLoginError('');
    
    if (!username || !password) {
      setLoginError('Please enter both username and password');
      return;
    }
    
    setIsAuthenticating(true);
    
    try {
      const response = await axios.post('/api/auth/login', { username, password });
      if (response.data && response.data.token) {
        onLoginSuccess(response.data);
      }
    } catch (err) {
      setLoginError(err.response?.data?.message || 'Login failed');
    } finally {
      setIsAuthenticating(false);
    }
  };
  
  return (
    <div className="auth-form-container">
      <h2 className="auth-title">Welcome Back</h2>
      {loginError && <div className="error-message">{loginError}</div>}
      
      <form onSubmit={handleLoginSubmit} className="auth-form">
        <div className="form-group">
          <label htmlFor="username">Username</label>
          <div className="input-with-icon">
            <i className="fas fa-user input-icon"></i>
            <input
              type="text"
              id="username"
              value={formData.username}
              onChange={handleInputChange}
              disabled={isAuthenticating}
              required
              placeholder="Enter your username"
              className="enhanced-input"
            />
          </div>
        </div>
        
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <div className="input-with-icon">
            <i className="fas fa-lock input-icon"></i>
            <input
              type="password"
              id="password"
              value={formData.password}
              onChange={handleInputChange}
              disabled={isAuthenticating}
              required
              placeholder="Enter your password"
              className="enhanced-input"
            />
          </div>
        </div>
        
        <button 
          type="submit" 
          className="btn-primary auth-button" 
          disabled={isAuthenticating}
        >
          {isAuthenticating ? 'Logging in...' : 'Login'}
        </button>
      </form>
      
      <p className="auth-switch">
        Don't have an account?{' '}
        <button 
          className="link-button" 
          onClick={onSwitchToRegister}
          disabled={isAuthenticating}
        >
          Register
        </button>
      </p>
    </div>
  );
};

export default LoginForm;
