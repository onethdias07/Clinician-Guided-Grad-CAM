import React, { useState } from 'react';
import axios from 'axios';

const RegistrationForm = ({ onRegistrationSuccess, onSwitchToLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [registrationError, setRegistrationError] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  
  const validateForm = () => {
    if (!username || !password || !confirmPassword) {
      return 'Please fill all fields';
    }
    if (password !== confirmPassword) {
      return 'Passwords do not match';
    }
    if (password.length < 6) {
      return 'Password must be at least 6 characters';
    }
    return null;
  };

  const handleRegistrationSubmit = async (e) => {
    e.preventDefault();
    
    const validationError = validateForm();
    if (validationError) {
      setRegistrationError(validationError);
      return;
    }
    
    setIsRegistering(true);
    setRegistrationError('');
    
    try {
      const response = await axios.post('/api/auth/register', {
        username,
        password,
        role: 'user'
      });
      
      if (response.status === 201) {
        onRegistrationSuccess();
      }
    } catch (error) {
      setRegistrationError(
        error.response?.status === 409 ? 'Username already exists' :
        error.response?.data?.message || 'Registration failed'
      );
    } finally {
      setIsRegistering(false);
    }
  };
  
  return (
    <div className="auth-form-container">
      <h2 className="auth-title">Create Account</h2>
      {registrationError && <div className="error-message">{registrationError}</div>}
      
      <form onSubmit={handleRegistrationSubmit} className="auth-form">
        <div className="form-group">
          <label htmlFor="reg-username">Username</label>
          <div className="input-with-icon">
            <i className="fas fa-user input-icon"></i>
            <input
              type="text"
              id="reg-username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={isRegistering}
              required
              placeholder="Choose a username"
              className="enhanced-input"
            />
          </div>
        </div>
        
        <div className="form-group">
          <label htmlFor="reg-password">Password</label>
          <div className="input-with-icon">
            <i className="fas fa-lock input-icon"></i>
            <input
              type="password"
              id="reg-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isRegistering}
              required
              placeholder="Create a password"
              className="enhanced-input"
            />
          </div>
        </div>
        
        <div className="form-group">
          <label htmlFor="confirm-password">Confirm Password</label>
          <div className="input-with-icon">
            <i className="fas fa-lock input-icon"></i>
            <input
              type="password"
              id="confirm-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              disabled={isRegistering}
              required
              placeholder="Confirm your password"
              className="enhanced-input"
            />
          </div>
        </div>
        
        <button 
          type="submit" 
          className="btn-primary auth-button" 
          disabled={isRegistering}
        >
          {isRegistering ? 'Registering...' : 'Register'}
        </button>
      </form>
      
      <p className="auth-switch">
        Already have an account?{' '}
        <button 
          className="link-button" 
          onClick={onSwitchToLogin}
          disabled={isRegistering}
        >
          Login
        </button>
      </p>
    </div>
  );
};

export default RegistrationForm;
