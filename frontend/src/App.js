import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  // State variables
  const [originalImage, setOriginalImage] = useState(null);
  const [gradCamImage, setGradCamImage] = useState(null);
  const [tbProbability, setTbProbability] = useState('--');
  const [tbLabel, setTbLabel] = useState(null); // Changed initial value to null for validation
  const [isDrawing, setIsDrawing] = useState(false);
  const [shapes, setShapes] = useState([]);
  const [currentShape, setCurrentShape] = useState([]);
  const [isCanvasVisible, setIsCanvasVisible] = useState(false);
  const [currentModel, setCurrentModel] = useState('tb_chest_xray_attention_best.pt');
  const [feedbackCount, setFeedbackCount] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState(null);
  
  // Add finetuning-related state
  const [isRefining, setIsRefining] = useState(false);
  const [finetuningProgress, setFinetuningProgress] = useState(0);
  const [finetuningStatus, setFinetuningStatus] = useState('');
  const [statusCheckInterval, setStatusCheckInterval] = useState(null);
  
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('tb_chest_xray_attention_best.pt');

  // Refs for canvas elements
  const displayCanvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Initialize canvas when gradCamImage changes
  useEffect(() => {
    if (gradCamImage) {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${gradCamImage}`;
      img.onload = () => {
        setupCanvas(img.width, img.height);
      };
    }
  }, [gradCamImage]);

  // Initial data load
  useEffect(() => {
    // Only update feedback count on initial load
    updateFeedbackCount();
    fetchCurrentModel();
    fetchAvailableModels();
  }, []);
  
  // Auto-hide success message after 5 seconds
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage(null);
      }, 5000);
      
      return () => clearTimeout(timer);
    }
  }, [successMessage]);

  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
      }
    };
  }, [statusCheckInterval]);

  // Handle file upload
  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Add file type validation
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
      
      setOriginalImage(response.data.original_image);
      setGradCamImage(response.data.grad_cam_image);
      setTbProbability(response.data.tb_probability);
      setIsLoading(false);
      setTbLabel(null); // Reset the label when uploading a new image
      
      // Clear any previous success message
      setSuccessMessage(null);
    } catch (error) {
      setIsLoading(false);
      alert('Error processing image. Please try a different file or try again later.');
    }
  };

  // Canvas setup
  const setupCanvas = (width, height) => {
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

    // Reset mask canvas
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, width, height);

    // Reset display canvas
    displayCtx.clearRect(0, 0, width, height);
    displayCtx.lineWidth = 2;
    displayCtx.strokeStyle = 'red';
    displayCtx.lineCap = 'round';

    setShapes([]);
    setCurrentShape([]);
    setIsDrawing(false);
  };

  // Drawing functions
  const toggleDraw = () => {
    if (!displayCanvasRef.current) return;
    
    const newVisibility = !isCanvasVisible;
    setIsCanvasVisible(newVisibility);
    
    // Reset shapes when hiding the canvas
    if (!newVisibility) {
      resetShapes();
    }
  };

  const handleMouseDown = (e) => {
    setIsDrawing(true);
    setCurrentShape([]);
    addPoint(e);
  };

  const handleMouseMove = (e) => {
    const rect = displayCanvasRef.current.getBoundingClientRect();
    
    if (!isDrawing) return;
    addPoint(e);
    redrawDisplay();
  };

  const handleMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false);
      finalizeShape();
    }
  };

  const handleMouseLeave = () => {
    if (isDrawing) {
      setIsDrawing(false);
      finalizeShape();
    }
  };

  // Fix for correct mouse coordinates
  const addPoint = (evt) => {
    if (!displayCanvasRef.current) return;
    
    const canvas = displayCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Calculate the scale factor between actual canvas dimensions and displayed size
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    // Get mouse position relative to canvas, then scale to actual canvas coordinates
    const x = (evt.clientX - rect.left) * scaleX;
    const y = (evt.clientY - rect.top) * scaleY;
    
    setCurrentShape(prev => [...prev, { x, y }]);
  };

  const redrawDisplay = () => {
    const displayCanvas = displayCanvasRef.current;
    if (!displayCanvas) return;
    
    const displayCtx = displayCanvas.getContext('2d');
    
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
    
    shapes.forEach(shape => drawPolyline(shape, displayCtx));
    
    if (currentShape.length > 1) {
      drawPolyline(currentShape, displayCtx);
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

  const finalizeShape = () => {
    if (currentShape.length < 2) {
      setCurrentShape([]);
      return;
    }
    
    setShapes(prevShapes => [...prevShapes, [...currentShape]]);
    fillShapeOnMask(currentShape);
    redrawDisplay();
    setCurrentShape([]);
  };

  const fillShapeOnMask = (shapePoints) => {
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

  const undoShape = () => {
    if (shapes.length === 0) return;
    
    const newShapes = [...shapes];
    newShapes.pop();
    setShapes(newShapes);
    rebuildMaskCanvas();
    redrawDisplay();
  };

  const resetShapes = () => {
    if (!maskCanvasRef.current || !displayCanvasRef.current) return;
    
    setShapes([]);
    setCurrentShape([]);
    setIsDrawing(false);
    
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
    
    shapes.forEach(s => fillShapeOnMask(s));
  };

  // Submit mask function with improved feedback
  const submitMask = async () => {
    if (shapes.length === 0) {
      alert('No shapes drawn to submit. Please draw on the image to indicate areas of interest.');
      return;
    }
    
    if (!originalImage) {
      alert('No original image data found.');
      return;
    }
    
    // Add validation for diagnosis selection
    if (tbLabel === null) {
      alert('Please select a diagnosis (TB Present or Normal) before submitting.');
      return;
    }
    
    setIsLoading(true);
    
    const maskDataURL = maskCanvasRef.current.toDataURL('image/png');
    const base64Mask = maskDataURL.split(',')[1];
    
    try {
      const response = await axios.post('/api/feedback', {
        image: originalImage,
        mask: base64Mask,
        label: tbLabel
      });
      
      setIsLoading(false);
      setSuccessMessage('Thank you! Your annotations have been successfully submitted.');
      
      // Reset drawing state
      resetShapes();
      setIsCanvasVisible(false);
      
      // Update feedback count
      updateFeedbackCount();
      
    } catch (error) {
      setIsLoading(false);
      alert('Error submitting annotations: ' + error.message);
    }
  };
  
  // Model refinement functions
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
  
  const runFinetuning = async () => {
    try {
      setIsRefining(true);
      setFinetuningProgress(0);
      setFinetuningStatus('Starting refinement process...');
      
      const response = await axios.post('/api/run_finetuning');
      
      if (response.data && response.data.success) {
        setFinetuningStatus(response.data.message);
        
        // Clear any existing interval
        if (statusCheckInterval) {
          clearInterval(statusCheckInterval);
        }
        
        // Start checking status
        const intervalId = setInterval(checkFinetuningStatus, 5000);
        setStatusCheckInterval(intervalId);
      } else {
        setFinetuningStatus(`Error: ${response.data?.message || 'Unknown error'}`);
        setIsRefining(false);
      }
    } catch (error) {
      setFinetuningStatus(`Error starting refinement: ${error.message}`);
      setIsRefining(false);
    }
  };
  
  const switchModel = async () => {
    try {
      setFinetuningStatus('Switching model...');
      
      const response = await axios.post('/api/switch_model', {
        model_name: selectedModel
      });
      
      if (response.data && response.data.success) {
        setFinetuningStatus(response.data.message);
        setCurrentModel(response.data.model_name);
      } else {
        setFinetuningStatus(`Error: ${response.data?.message || 'Unknown error'}`);
      }
    } catch (error) {
      setFinetuningStatus(`Error switching model: ${error.message}`);
    }
  };
  
  const checkFinetuningStatus = async () => {
    try {
      const response = await axios.get('/api/finetuning_status');
      
      setFinetuningStatus(response.data.message);
      
      if (response.data.current_epoch && response.data.total_epochs) {
        const progress = (response.data.current_epoch / response.data.total_epochs) * 100;
        setFinetuningProgress(progress);
      }
      
      if (!response.data.running) {
        setIsRefining(false);
        setFinetuningProgress(100);
        
        if (statusCheckInterval) {
          clearInterval(statusCheckInterval);
          setStatusCheckInterval(null);
        }
        
        // Update feedback count after finetuning completes
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
        // Set available models
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

  // Handle model selection change
  const handleModelSelectionChange = (e) => {
    setSelectedModel(e.target.value);
  };

  // Helper functions
  const formatProbability = (prob) => {
    if (prob === '--') return '--';
    const numProb = parseFloat(prob);
    return numProb.toFixed(1);
  };
  
  const getProbabilityColor = (prob) => {
    if (prob === '--') return 'var(--gray-dark)';
    const numProb = parseFloat(prob);
    if (numProb > 75) return 'var(--warning)';  // Adjusted thresholds
    if (numProb > 40) return 'var(--accent)';
    return 'var(--success)';
  };
  
  const getProbabilityClass = (prob) => {
    if (prob === '--') return '';
    const numProb = parseFloat(prob);
    return numProb > 75 ? 'high-probability' : numProb > 40 ? 'medium-probability' : 'low-probability';
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="header">
        <div className="header-container">
          <h1 className="header-title">Clinician Guided Grad-CAM</h1>
        </div>
      </header>
      
      <div className="main-container">
        {/* File Upload Card */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Upload X-ray Image</h2>
          </div>
          <div className="file-upload">
            <p>Select a chest X-ray image to analyze</p>
            <input 
              type="file" 
              accept="image/jpeg,image/png,image/bmp,image/tiff" 
              onChange={handleFileUpload}
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
        
        {/* Success Message */}
        {successMessage && (
          <div className="success-status">
            <h4>✓ Success!</h4>
            <p>{successMessage}</p>
          </div>
        )}
        
        {/* Images Section */}
        {originalImage && (
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Analysis Results</h2>
            </div>
            
            {/* TB Probability Display */}
            <div className="result">
              <div className="probability-display">
                <span className="probability-label">Tuberculosis Probability:</span>
                <span 
                  className={`probability-value ${getProbabilityClass(tbProbability)}`}
                  style={{ color: getProbabilityColor(tbProbability) }}
                >
                  {formatProbability(tbProbability)}%
                </span>
              </div>
            </div>
            
            {/* Images */}
            <div className="image-section">
              {/* Original image */}
              <div className="img-container">
                <div className="img-header">Original X-ray</div>
                <div className="img-content">
                  <img 
                    src={`data:image/jpeg;base64,${originalImage}`} 
                    alt="Original X-ray" 
                  />
                </div>
              </div>
              
              {/* Grad-CAM overlay */}
              <div className="img-container">
                <div className="img-header">
                  Grad-CAM Analysis
                  {gradCamImage && (
                    <button
                      className="btn-secondary"
                      style={{padding: '6px 12px', fontSize: '0.9rem'}}
                      onClick={toggleDraw}
                    >
                      {isCanvasVisible ? 'Hide Drawing Tool' : 'Draw Annotations'}
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
                        style={{ display: isCanvasVisible ? 'block' : 'none' }}
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
            
            {/* Moved Label Selection above Drawing Tools for better visibility */}
            {gradCamImage && isCanvasVisible && (
              <div className="label-section" style={{ marginBottom: '25px' }}>
                <h3 className="label-title">Select Diagnosis (Required):</h3>
                <div className="label-row">
                  <label className="radio-label">
                    <input 
                      type="radio" 
                      name="tb_label" 
                      value="TB"
                      checked={tbLabel === 'TB'}
                      onChange={() => setTbLabel('TB')}
                    />
                    TB Present
                  </label>
                  <label className="radio-label">
                    <input 
                      type="radio" 
                      name="tb_label" 
                      value="Normal"
                      checked={tbLabel === 'Normal'}
                      onChange={() => setTbLabel('Normal')}
                    />
                    Normal
                  </label>
                </div>
              </div>
            )}
            
            {/* Drawing Tools */}
            {gradCamImage && isCanvasVisible && (
              <div className="drawing-tools">
                <button 
                  className="btn-secondary" 
                  onClick={undoShape} 
                  disabled={shapes.length === 0}
                >
                  Undo Last Shape
                </button>
                <button 
                  className="btn-secondary" 
                  onClick={resetShapes} 
                  disabled={shapes.length === 0}
                >
                  Clear All
                </button>
                <button 
                  className="btn-primary" 
                  onClick={submitMask} 
                  disabled={shapes.length === 0 || tbLabel === null}
                  title={tbLabel === null ? "Please select a diagnosis before submitting" : ""}
                >
                  Submit Annotations
                </button>
              </div>
            )}
            
            {/* Drawing Instructions */}
            {gradCamImage && isCanvasVisible && (
              <div className="text-muted" style={{ textAlign: 'center', marginTop: '15px', fontStyle: 'italic' }}>
                Draw shapes around areas of interest in the image. Click and drag to create shapes.
              </div>
            )}
          </div>
        )}
        
        {/* Model Refinement Section */}
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
              onClick={runFinetuning} 
              disabled={isRefining || feedbackCount === 0}
              title={feedbackCount === 0 ? "Need feedback data to refine model" : ""}
            >
              Run Offline Refinement
            </button>
            
            <div className="model-selector">
              <select 
                value={selectedModel}
                onChange={handleModelSelectionChange}
                disabled={isRefining || availableModels.length === 0}
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
                disabled={isRefining || selectedModel === currentModel}
                title={selectedModel === currentModel ? "Already using this model" : ""}
              >
                Switch Model
              </button>
            </div>
          </div>
          
          {/* Progress bar for finetuning */}
          {(isRefining || finetuningStatus) && (
            <div className="refinement-status">
              {isRefining && (
                <div className="progress-container">
                  <div className="progress-label">Refinement progress:</div>
                  <div className="progress-bar-container">
                    <div 
                      className="progress-bar" 
                      style={{ width: `${finetuningProgress}%` }}
                    ></div>
                  </div>
                  <div className="progress-text">{Math.round(finetuningProgress)}%</div>
                </div>
              )}
              <div className="status-message">{finetuningStatus}</div>
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
      
      {/* Loading overlay */}
      {isLoading && (
        <div className="spinner-overlay">
          <div className="spinner"></div>
        </div>
      )}
      
      {/* Footer */}
      <footer className="footer">
        <div>© 2025 Clinician-Guided Grad-CAM | Tuberculosis Detection AI</div>
      </footer>
    </div>
  );
}

export default App;
