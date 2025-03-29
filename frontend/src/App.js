import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import LoginForm from './components/LoginForm';
import RegistrationForm from './components/RegistrationForm';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authToken, setAuthToken] = useState('');
  const [username, setUsername] = useState('');
  const [userRole, setUserRole] = useState('');
  const [showLoginForm, setShowLoginForm] = useState(true);
  const [showRegistrationForm, setShowRegistrationForm] = useState(false);
  
  const [originalXrayImage, setOriginalXrayImage] = useState(null);
  const [gradCamImage, setGradCamImage] = useState(null);
  const [tuberculosisProbability, setTuberculosisProbability] = useState('--');
  const [tuberculosisDiagnosis, setTuberculosisDiagnosis] = useState(null);
  const [isDrawingAnnotation, setIsDrawingAnnotation] = useState(false);
  const [annotationShapes, setAnnotationShapes] = useState([]);
  const [currentAnnotationShape, setCurrentAnnotationShape] = useState([]);
  const [isAnnotationToolVisible, setIsAnnotationToolVisible] = useState(false);
  const [currentModel, setCurrentModel] = useState('tb_chest_xray_attention_best.pt');
  const [feedbackCount, setFeedbackCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackSubmissionSuccess, setFeedbackSubmissionSuccess] = useState(null);
  
  const [isRefiningModel, setIsRefiningModel] = useState(false);
  const [modelRefinementProgress, setModelRefinementProgress] = useState(0);
  const [modelRefinementStatus, setModelRefinementStatus] = useState('');
  const [statusCheckInterval, setStatusCheckInterval] = useState(null);
  
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('tb_chest_xray_attention_best.pt');

  const displayCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    const token = localStorage.getItem('authToken');
    const storedUsername = localStorage.getItem('username');
    const storedUserRole = localStorage.getItem('userRole');
    
    if (token) {
      setAuthToken(token);
      setUsername(storedUsername || '');
      setUserRole(storedUserRole || '');
      verifyToken(token);
    }
    
    axios.interceptors.request.use(
      config => {
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers['Authorization'] = `Bearer ${token}`;
        }
        return config;
      },
      error => {
        return Promise.reject(error);
      }
    );
    
    axios.interceptors.response.use(
      response => response,
      error => {
        if (error.response && error.response.status === 401) {
          handleLogout();
        }
        return Promise.reject(error);
      }
    );
  }, []);
  
  const verifyToken = async (token) => {
    try {
      const response = await axios.get('/api/auth/verify', {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      
      if (response.data && response.data.authenticated) {
        setIsAuthenticated(true);
        setUsername(response.data.username);
        setUserRole(response.data.role);
        localStorage.setItem('username', response.data.username);
        localStorage.setItem('userRole', response.data.role);
      } else {
        handleLogout();
      }
    } catch (error) {
      console.error('Token verification failed:', error);
      handleLogout();
    }
  };
  
  const handleLoginSuccess = (data) => {
    setAuthToken(data.token);
    setIsAuthenticated(true);
    setUsername(data.username);
    setUserRole(data.role);
    localStorage.setItem('authToken', data.token);
    localStorage.setItem('username', data.username);
    localStorage.setItem('userRole', data.role);
    setShowLoginForm(false);
    setShowRegistrationForm(false);
  };
  
  const handleLogout = () => {
    setAuthToken('');
    setIsAuthenticated(false);
    setUsername('');
    setUserRole('');
    localStorage.removeItem('authToken');
    localStorage.removeItem('username');
    localStorage.removeItem('userRole');
    setShowLoginForm(true);
  };
  
  const toggleAuthenticationView = () => {
    setShowLoginForm(!showLoginForm);
    setShowRegistrationForm(!showRegistrationForm);
  };

  useEffect(() => {
    if (gradCamImage) {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${gradCamImage}`;
      img.onload = () => {
        setupAnnotationCanvas(img.width, img.height);
      };
    }
  }, [gradCamImage]);

  useEffect(() => {
    if (isAuthenticated) {
      updateFeedbackCount();
      fetchCurrentModel();
      fetchAvailableModels();
    }
  }, [isAuthenticated]);
  
  useEffect(() => {
    if (feedbackSubmissionSuccess) {
      const timer = setTimeout(() => {
        setFeedbackSubmissionSuccess(null);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, [feedbackSubmissionSuccess]);

  useEffect(() => {
    return () => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
      }
    };
  }, [statusCheckInterval]);

  const handleXrayImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
      alert('Please upload a valid image file (JPEG, PNG, BMP, or TIFF)');
      return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    setIsLoading(true);
    
    try {
      const response = await axios.post('/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      const { original_image, grad_cam_image, tb_probability } = response.data;
      
      setOriginalXrayImage(original_image);
      setGradCamImage(grad_cam_image);
      setTuberculosisProbability(tb_probability);
      setTuberculosisDiagnosis(null);
      setFeedbackSubmissionSuccess(null);
    } catch (error) {
      alert('Error processing image. Please try a different file or try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const setupAnnotationCanvas = (width, height) => {
    if (!displayCanvasRef.current || !maskCanvasRef.current) return;

    const displayCanvas = displayCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const displayCtx = displayCanvas.getContext('2d');
    const maskCtx = maskCanvas.getContext('2d');

    displayCanvas.width = width;
    displayCanvas.height = height;
    maskCanvas.width = width;
    maskCanvas.height = height;

    displayCanvas.style.width = '100%';
    displayCanvas.style.height = 'auto';

    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, width, height);

    displayCtx.clearRect(0, 0, width, height);
    displayCtx.lineWidth = 2;
    displayCtx.strokeStyle = 'red';
    displayCtx.lineCap = 'round';

    setAnnotationShapes([]);
    setCurrentAnnotationShape([]);
    setIsDrawingAnnotation(false);
  };

  const toggleAnnotationTool = () => {
    if (!displayCanvasRef.current) return;
    
    const newVisibility = !isAnnotationToolVisible;
    setIsAnnotationToolVisible(newVisibility);
    
    if (!newVisibility) {
      resetAnnotations();
    }
  };

  const handleMouseDown = (e) => {
    setIsDrawingAnnotation(true);
    setCurrentAnnotationShape([]);
    addAnnotationPoint(e);
  };

  const handleMouseMove = (e) => {
    const rect = displayCanvasRef.current.getBoundingClientRect();
    
    if (!isDrawingAnnotation) return;
    addAnnotationPoint(e);
    redrawDisplayCanvas();
  };

  const handleMouseUp = () => {
    if (isDrawingAnnotation) {
      setIsDrawingAnnotation(false);
      finalizeAnnotationShape();
    }
  };

  const handleMouseLeave = () => {
    if (isDrawingAnnotation) {
      setIsDrawingAnnotation(false);
      finalizeAnnotationShape();
    }
  };

  const addAnnotationPoint = (evt) => {
    if (!displayCanvasRef.current) return;
    
    const canvas = displayCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (evt.clientX - rect.left) * scaleX;
    const y = (evt.clientY - rect.top) * scaleY;
    
    setCurrentAnnotationShape(prev => [...prev, { x, y }]);
  };

  const redrawDisplayCanvas = () => {
    const displayCanvas = displayCanvasRef.current;
    if (!displayCanvas) return;
    
    const displayCtx = displayCanvas.getContext('2d');
    
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
    
    annotationShapes.forEach(shape => drawPolyline(shape, displayCtx));
    
    if (currentAnnotationShape.length > 1) {
      drawPolyline(currentAnnotationShape, displayCtx);
    }
  };

  const drawPolyline = (points, ctx) => {
    if (!points || points.length < 2) return;
    
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();
  };

  const finalizeAnnotationShape = () => {
    if (currentAnnotationShape.length < 2) {
      setCurrentAnnotationShape([]);
      return;
    }
    
    setAnnotationShapes(prevShapes => [...prevShapes, [...currentAnnotationShape]]);
    fillAnnotationShapeOnMask(currentAnnotationShape);
    redrawDisplayCanvas();
    setCurrentAnnotationShape([]);
  };

  const fillAnnotationShapeOnMask = (shapePoints) => {
    if (!shapePoints || shapePoints.length < 3 || !maskCanvasRef.current) return;
    
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    
    maskCtx.beginPath();
    maskCtx.fillStyle = 'white';
    maskCtx.moveTo(shapePoints[0].x, shapePoints[0].y);
    
    for (let i = 1; i < shapePoints.length; i++) {
      maskCtx.lineTo(shapePoints[i].x, shapePoints[i].y);
    }
    
    maskCtx.closePath();
    maskCtx.fill();
  };

  const undoAnnotationShape = () => {
    if (annotationShapes.length === 0) return;
    
    const newShapes = [...annotationShapes];
    newShapes.pop();
    setAnnotationShapes(newShapes);
    rebuildMaskCanvas();
    redrawDisplayCanvas();
  };

  const resetAnnotations = () => {
    if (!maskCanvasRef.current || !displayCanvasRef.current) return;
    
    setAnnotationShapes([]);
    setCurrentAnnotationShape([]);
    setIsDrawingAnnotation(false);
    
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    
    const displayCanvas = displayCanvasRef.current;
    const displayCtx = displayCanvas.getContext('2d');
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  };

  const rebuildMaskCanvas = () => {
    if (!maskCanvasRef.current) return;
    
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    
    annotationShapes.forEach(s => fillAnnotationShapeOnMask(s));
  };

  const submitAnnotationMask = async () => {
    if (annotationShapes.length === 0) {
      alert('No shapes drawn to submit. Please draw on the image to indicate areas of interest.');
      return;
    }
    
    if (!originalXrayImage) {
      alert('No original image data found.');
      return;
    }
    
    if (tuberculosisDiagnosis === null) {
      alert('Please select a diagnosis (TB Present or Normal) before submitting.');
      return;
    }
    
    setIsLoading(true);
    
    const maskDataURL = maskCanvasRef.current.toDataURL('image/png');
    const base64Mask = maskDataURL.split(',')[1];
    
    try {
      await axios.post('/api/feedback', {
        image: originalXrayImage,
        mask: base64Mask,
        label: tuberculosisDiagnosis
      });
      
      setFeedbackSubmissionSuccess('Thank you! Your annotations have been successfully submitted.');
      resetAnnotations();
      setIsAnnotationToolVisible(false);
      updateFeedbackCount();
      
    } catch (error) {
      alert('Error submitting annotations: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const fetchCurrentModel = async () => {
    try {
      const response = await axios.get('/api/current_model');
      if (response.data && response.data.model_name) {
        setCurrentModel(response.data.model_name);
      }
    } catch (error) {
      console.error('Error fetching current model:', error);
    }
  };
  
  const updateFeedbackCount = async () => {
    try {
      const response = await axios.get('/api/feedback_count');
      setFeedbackCount(response.data.count);
    } catch (error) {
      console.error('Error updating feedback count:', error);
    }
  };
  
  const runModelRefinement = async () => {
    try {
      setIsRefiningModel(true);
      setModelRefinementProgress(0);
      setModelRefinementStatus('Starting refinement process...');
      
      const response = await axios.post('/api/run_finetuning');
      
      if (response.data && response.data.success) {
        setModelRefinementStatus(response.data.message);
        
        if (statusCheckInterval) {
          clearInterval(statusCheckInterval);
        }
        
        const intervalId = setInterval(checkModelRefinementStatus, 5000);
        setStatusCheckInterval(intervalId);
      } else {
        setModelRefinementStatus(`Error: ${response.data?.message || 'Unknown error'}`);
        setIsRefiningModel(false);
      }
    } catch (error) {
      setModelRefinementStatus(`Error starting refinement: ${error.message}`);
      setIsRefiningModel(false);
    }
  };
  
  const switchModel = async () => {
    try {
      setModelRefinementStatus('Switching model...');
      
      const response = await axios.post('/api/switch_model', {
        model_name: selectedModel
      });
      
      if (response.data && response.data.success) {
        setModelRefinementStatus(response.data.message);
        setCurrentModel(response.data.model_name);
      } else {
        setModelRefinementStatus(`Error: ${response.data?.message || 'Unknown error'}`);
      }
    } catch (error) {
      setModelRefinementStatus(`Error switching model: ${error.message}`);
    }
  };
  
  const checkModelRefinementStatus = async () => {
    try {
      const response = await axios.get('/api/finetuning_status');
      
      setModelRefinementStatus(response.data.message);
      
      const { current_epoch, total_epochs, running } = response.data;
      
      if (current_epoch && total_epochs) {
        const progress = (current_epoch / total_epochs) * 100;
        setModelRefinementProgress(progress);
      }
      
      if (!running) {
        setIsRefiningModel(false);
        setModelRefinementProgress(100);
        
        if (statusCheckInterval) {
          clearInterval(statusCheckInterval);
          setStatusCheckInterval(null);
        }
        
        updateFeedbackCount();
      }
    } catch (error) {
      console.error('Error checking finetuning status:', error);
    }
  };

  const fetchAvailableModels = async () => {
    try {
      const response = await axios.get('/api/available_models');
      
      if (response.data) {
        const models = [
          response.data.default_model,
          ...(response.data.refined_models || [])
        ];
        
        setAvailableModels(models);
      }
    } catch (error) {
      console.error('Error fetching available models:', error);
    }
  };

  const handleModelSelectionChange = (e) => {
    setSelectedModel(e.target.value);
  };

  const formatProbabilityDisplay = (prob) => {
    if (prob === '--') {
      return {
        display: '--',
        color: 'var(--gray-dark)',
        class: ''
      };
    }
    
    const numProb = parseFloat(prob);
    return {
      display: numProb.toFixed(1),
      color: numProb > 75 ? 'var(--warning)' : 
             numProb > 40 ? 'var(--accent)' : 'var(--success)',
      class: numProb > 75 ? 'high-probability' : 
             numProb > 40 ? 'medium-probability' : 'low-probability'
    };
  };

  return (
    <div className="App">
      <header className="header">
        <div className="header-container">
          <h1 className="header-title">Clinician Guided Grad-CAM</h1>
          {isAuthenticated && (
            <div className="user-info">
              <span className="username">
                {username} ({userRole})
              </span>
              <button 
                className="btn-secondary logout-btn" 
                onClick={handleLogout}
              >
                Logout
              </button>
            </div>
          )}
        </div>
      </header>
      
      {!isAuthenticated && (
        <div className="auth-container">
          {showLoginForm && (
            <LoginForm 
              onLoginSuccess={handleLoginSuccess} 
              onSwitchToRegister={toggleAuthenticationView}
            />
          )}
          
          {showRegistrationForm && (
            <RegistrationForm 
              onRegistrationSuccess={() => setShowLoginForm(true)} 
              onSwitchToLogin={toggleAuthenticationView}
            />
          )}
        </div>
      )}
      
      {isAuthenticated && (
        <div className="main-container">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Upload X-ray Image</h2>
            </div>
            <div className="file-upload">
              <p>Select a chest X-ray image to analyze</p>
              <input 
                type="file" 
                accept="image/jpeg,image/png,image/bmp,image/tiff" 
                onChange={handleXrayImageUpload}
                disabled={isLoading}
                ref={fileInputRef}
              />
              <button 
                className="file-upload-btn"
                onClick={() => fileInputRef.current.click()}
                disabled={isLoading}
              >
                {isLoading ? 'Processing...' : 'Select Image'}
              </button>
            </div>
          </div>
          
          {feedbackSubmissionSuccess && (
            <div className="success-status">
              <h4>✓ Success!</h4>
              <p>{feedbackSubmissionSuccess}</p>
            </div>
          )}
          
          {originalXrayImage && (
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Analysis Results</h2>
              </div>
              
              <div className="result">
                <div className="probability-display">
                  <span className="probability-label">Tuberculosis Probability:</span>
                  <span 
                    className={`probability-value ${formatProbabilityDisplay(tuberculosisProbability).class}`}
                    style={{ color: formatProbabilityDisplay(tuberculosisProbability).color }}
                  >
                    {formatProbabilityDisplay(tuberculosisProbability).display}%
                  </span>
                </div>
              </div>
              
              <div className="image-section">
                <div className="img-container">
                  <div className="img-header">Original X-ray</div>
                  <div className="img-content">
                    <img 
                      src={`data:image/jpeg;base64,${originalXrayImage}`} 
                      alt="Original X-ray" 
                    />
                  </div>
                </div>
                
                <div className="img-container">
                  <div className="img-header">
                    Grad-CAM Analysis
                    {gradCamImage && (
                      <button
                        className="btn-secondary"
                        style={{padding: '6px 12px', fontSize: '0.9rem'}}
                        onClick={toggleAnnotationTool}
                      >
                        {isAnnotationToolVisible ? 'Hide Drawing Tool' : 'Draw Annotations'}
                      </button>
                    )}
                  </div>
                  <div className="img-content">
                    {gradCamImage && (
                      <>
                        <img 
                          src={`data:image/jpeg;base64,${gradCamImage}`} 
                          alt="Grad-CAM Overlay" 
                        />
                        <canvas 
                          ref={displayCanvasRef}
                          id="displayCanvas"
                          style={{ display: isAnnotationToolVisible ? 'block' : 'none' }}
                          onMouseDown={handleMouseDown}
                          onMouseMove={handleMouseMove}
                          onMouseUp={handleMouseUp}
                          onMouseLeave={handleMouseLeave}
                        ></canvas>
                        <canvas 
                          ref={maskCanvasRef}
                          id="maskCanvas"
                          style={{ display: 'none' }}
                        ></canvas>
                      </>
                    )}
                  </div>
                </div>
              </div>
              
              {gradCamImage && isAnnotationToolVisible && (
                <div className="label-section" style={{ marginBottom: '25px' }}>
                  <h3 className="label-title">Select Diagnosis (Required):</h3>
                  <div className="label-row">
                    <label className="radio-label">
                      <input 
                        type="radio" 
                        name="tb_label" 
                        value="TB"
                        checked={tuberculosisDiagnosis === 'TB'}
                        onChange={() => setTuberculosisDiagnosis('TB')}
                      />
                      TB Present
                    </label>
                    <label className="radio-label">
                      <input 
                        type="radio" 
                        name="tb_label" 
                        value="Normal"
                        checked={tuberculosisDiagnosis === 'Normal'}
                        onChange={() => setTuberculosisDiagnosis('Normal')}
                      />
                      Normal
                    </label>
                  </div>
                </div>
              )}
              
              {gradCamImage && isAnnotationToolVisible && (
                <div className="drawing-tools">
                  <button 
                    className="btn-secondary" 
                    onClick={undoAnnotationShape} 
                    disabled={annotationShapes.length === 0}
                  >
                    Undo Last Shape
                  </button>
                  <button 
                    className="btn-secondary" 
                    onClick={resetAnnotations} 
                    disabled={annotationShapes.length === 0}
                  >
                    Clear All
                  </button>
                  <button 
                    className="btn-primary" 
                    onClick={submitAnnotationMask} 
                    disabled={annotationShapes.length === 0 || tuberculosisDiagnosis === null}
                    title={tuberculosisDiagnosis === null ? "Please select a diagnosis before submitting" : ""}
                  >
                    Submit Annotations
                  </button>
                </div>
              )}
              
              {gradCamImage && isAnnotationToolVisible && (
                <div className="text-muted" style={{ textAlign: 'center', marginTop: '15px', fontStyle: 'italic' }}>
                  Draw shapes around areas of interest in the image. Click and drag to create shapes.
                </div>
              )}
            </div>
          )}
          
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Model Management</h2>
            </div>
            
            <div className="model-info">
              <div className="model-info-item">
                <div className="model-info-label">Current Model</div>
                <div className="model-info-value">{currentModel}</div>
              </div>
              <div className="model-info-item">
                <div className="model-info-label">Annotations Collected</div>
                <div className="model-info-value">{feedbackCount}</div>
              </div>
            </div>
            
            <div className="model-actions">
              <button 
                className="btn-primary" 
                onClick={runModelRefinement} 
                disabled={isRefiningModel || feedbackCount === 0}
                title={feedbackCount === 0 ? "Need feedback data to refine model" : ""}
              >
                Run Offline Refinement
              </button>
              
              <div className="model-selector">
                <select 
                  value={selectedModel}
                  onChange={handleModelSelectionChange}
                  disabled={isRefiningModel || availableModels.length === 0}
                  className="model-dropdown"
                >
                  {availableModels.map((model, index) => (
                    <option key={index} value={model}>
                      {model === 'tb_chest_xray_attention_best.pt' ? 'Default Model' : model}
                    </option>
                  ))}
                </select>
                
                <button 
                  className="btn-secondary" 
                  onClick={switchModel}
                  disabled={isRefiningModel || selectedModel === currentModel}
                  title={selectedModel === currentModel ? "Already using this model" : ""}
                >
                  Switch Model
                </button>
              </div>
            </div>
            
            {(isRefiningModel || modelRefinementStatus) && (
              <div className="refinement-status">
                {isRefiningModel && (
                  <div className="progress-container">
                    <div className="progress-label">Refinement progress:</div>
                    <div className="progress-bar-container">
                      <div 
                        className="progress-bar" 
                        style={{ width: `${modelRefinementProgress}%` }}
                      ></div>
                    </div>
                    <div className="progress-text">{Math.round(modelRefinementProgress)}%</div>
                  </div>
                )}
                <div className="status-message">{modelRefinementStatus}</div>
              </div>
            )}
            
            <div className="help-text">
              <p>
                <strong>Offline Refinement</strong> - Refine the model using clinician's feedback data
              </p>
              <p>
                <strong>Switch Model</strong> - Use the latest refined model for predictions
              </p>
            </div>
          </div>
        </div>
      )}
      
      {isLoading && (
        <div className="spinner-overlay">
          <div className="spinner"></div>
        </div>
      )}
      
      <footer className="footer">
        <div>© 2025 Clinician-Guided Grad-CAM | Tuberculosis Detection AI</div>
      </footer>
    </div>
  );
}

export default App;
