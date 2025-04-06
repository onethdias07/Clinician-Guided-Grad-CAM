## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- pip and npm package managers

### Backend Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Start the Flask server:
```bash 
python app.py
```

The backend will run on `http://localhost:8000`

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

2. Start the development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

## Usage

### Default Login Credentials

- Admin User:
  - Username: `admin`
  - Password: `admin`

- Regular User:  
  - Username: `user`
  - Password: `password`

### Basic Workflow

1. Log in using provided credentials
2. Upload a chest X-ray image for analysis
3. View TB probability and Grad-CAM visualization
4. Draw annotations on regions of interest (if needed)
5. Submit feedback with corrected diagnosis
6. Run model refinement with collected feedback

### Model Refinement

1. Collect sufficient expert feedback (minimum recommended: 100 samples)
2. Click "Run Offline Refinement" to start the process
3. Monitor refinement progress in real-time
4. Switch to refined model once complete

## Security

- JWT-based authentication
- Role-based access control
- Secure file handling
- Input validation and sanitization

## Acknowledgments

- PyTorch for deep learning framework
- React for frontend development
- Flask for backend API
