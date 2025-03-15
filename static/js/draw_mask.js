const gradcamImg = document.getElementById("gradcamImg");
const displayCanvas = document.getElementById("displayCanvas");
const displayCtx = displayCanvas ? displayCanvas.getContext("2d") : null;
const maskCanvas = document.getElementById("maskCanvas");
const maskCtx = maskCanvas ? maskCanvas.getContext("2d") : null;

const toggleDrawBtn = document.getElementById("toggleDrawBtn");
const undoBtn = document.getElementById("undoBtn");
const resetBtn = document.getElementById("resetBtn");
const submitMaskBtn = document.getElementById("submitMaskBtn");

const originalImageElem = document.getElementById("originalImageData");
const originalImageBase64 = originalImageElem ? originalImageElem.textContent : "";

function getTbLabel() {
    const radios = document.querySelectorAll('input[name="tb_label"]');
    for (let r of radios) {
        if (r.checked) {
            return r.value;
        }
    }
    return "Unknown";
}

let shapes = [];
let currentShape = [];
let drawing = false;

if (gradcamImg) {
    gradcamImg.onload = () => {
        setupCanvas(gradcamImg.width, gradcamImg.height);
    };
    if (gradcamImg.complete) {
        setupCanvas(gradcamImg.width, gradcamImg.height);
    }
}

function setupCanvas(width, height) {
    if (!displayCanvas || !maskCanvas) return;

    displayCanvas.width = width;
    displayCanvas.height = height;
    displayCanvas.style.width = width + "px";
    displayCanvas.style.height = height + "px";
    displayCanvas.style.display = "none";

    maskCanvas.width = width;
    maskCanvas.height = height;

    maskCtx.fillStyle = "black";
    maskCtx.fillRect(0, 0, width, height);

    displayCtx.clearRect(0, 0, width, height);
    displayCtx.lineWidth = 2;
    displayCtx.strokeStyle = "red";
    displayCtx.lineCap = "round";

    shapes = [];
    currentShape = [];
    drawing = false;

    toggleDrawBtn.disabled = false;
    undoBtn.disabled = true;
    resetBtn.disabled = true;
    submitMaskBtn.disabled = true;
}

if (toggleDrawBtn) {
    toggleDrawBtn.addEventListener("click", () => {
        if (!displayCanvas) return;
        if (displayCanvas.style.display === "none") {
            displayCanvas.style.display = "block";
        } else {
            displayCanvas.style.display = "none";
        }
    });
}

if (displayCanvas) {
    displayCanvas.addEventListener("mousedown", (e) => {
        drawing = true;
        currentShape = [];
        addPoint(e);
    });
    displayCanvas.addEventListener("mousemove", (e) => {
        if (!drawing) return;
        addPoint(e);
        redrawDisplay();
    });
    displayCanvas.addEventListener("mouseup", () => {
        if (drawing) {
            drawing = false;
            finalizeShape();
        }
    });
    displayCanvas.addEventListener("mouseleave", () => {
        if (drawing) {
            drawing = false;
            finalizeShape();
        }
    });
}

function addPoint(evt) {
    const rect = displayCanvas.getBoundingClientRect();
    const x = evt.clientX - rect.left;
    const y = evt.clientY - rect.top;
    currentShape.push({ x, y });
}

function redrawDisplay() {
    displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
    shapes.forEach(shape => drawPolyline(shape));
    if (currentShape.length > 1) {
        drawPolyline(currentShape);
    }
}

function drawPolyline(points) {
    displayCtx.beginPath();
    displayCtx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
        displayCtx.lineTo(points[i].x, points[i].y);
    }
    displayCtx.stroke();
}

function finalizeShape() {
    if (currentShape.length < 2) {
        currentShape = [];
        return;
    }
    shapes.push([...currentShape]);
    fillShapeOnMask(currentShape);

    undoBtn.disabled = false;
    resetBtn.disabled = false;
    submitMaskBtn.disabled = false;

    redrawDisplay();
    currentShape = [];
}

function fillShapeOnMask(shapePoints) {
    maskCtx.beginPath();
    maskCtx.fillStyle = "white";
    maskCtx.moveTo(shapePoints[0].x, shapePoints[0].y);
    for (let i = 1; i < shapePoints.length; i++) {
        maskCtx.lineTo(shapePoints[i].x, shapePoints[i].y);
    }
    maskCtx.closePath();
    maskCtx.fill();
}

if (undoBtn) {
    undoBtn.addEventListener("click", () => {
        if (shapes.length === 0) return;
        shapes.pop();
        rebuildMaskCanvas();
        redrawDisplay();

        if (shapes.length === 0) {
            undoBtn.disabled = true;
            resetBtn.disabled = true;
            submitMaskBtn.disabled = true;
        }
    });
}

if (resetBtn) {
    resetBtn.addEventListener("click", () => {
        shapes = [];
        currentShape = [];
        drawing = false;
        maskCtx.fillStyle = "black";
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
        displayCtx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);

        undoBtn.disabled = true;
        resetBtn.disabled = true;
        submitMaskBtn.disabled = true;
    });
};

function rebuildMaskCanvas() {
    maskCtx.fillStyle = "black";
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    shapes.forEach(s => fillShapeOnMask(s));
}

if (submitMaskBtn) {
    submitMaskBtn.addEventListener("click", async () => {
        if (shapes.length === 0) {
            alert("No shapes drawn to submit.");
            return;
        }
        const maskDataURL = maskCanvas.toDataURL("image/png");
        const base64Mask = maskDataURL.split(",")[1];

        if (!originalImageBase64) {
            alert("No original image data found.");
            return;
        }

        const selectedLabel = getTbLabel();

        try {
            let resp = await fetch("/submit_mask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: originalImageBase64,
                    mask: base64Mask,
                    label: selectedLabel
                })
            });
            if (!resp.ok) {
                let txt = await resp.text();
                alert(`Error /submit_mask: ${resp.status} ${txt}`);
                return;
            }
            let data = await resp.json();
            console.log("[/submit_mask] success:", data);
            alert("Mask + image + label submitted successfully!");
        } catch (err) {
            alert("Exception in /submit_mask: " + err.message);
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const runFinetuningBtn = document.getElementById('runFinetuningBtn');
    const switchModelBtn = document.getElementById('switchModelBtn');
    const finetuningStatus = document.getElementById('finetuning-status');
    const currentModelSpan = document.getElementById('current-model');
    const feedbackCountSpan = document.getElementById('feedback-count');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    let statusCheckInterval = null;
    let startTime = null;
    const expectedDuration = 120000;
    
    if (runFinetuningBtn) {
        runFinetuningBtn.addEventListener('click', function() {
            runFinetuningBtn.disabled = true;
            finetuningStatus.textContent = "Starting finetuning process...";
            
            startTime = Date.now();
            progressBar.style.width = "0%";
            progressText.textContent = "0%";
            progressContainer.style.display = "block";
            
            fetch('/run_finetuning', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    finetuningStatus.textContent = data.message;
                    
                    if (statusCheckInterval) {
                        clearInterval(statusCheckInterval);
                    }
                    
                    statusCheckInterval = setInterval(checkFinetuningStatus, 5000);
                    
                    updateProgressBar();
                } else {
                    finetuningStatus.textContent = "Error: " + data.message;
                    runFinetuningBtn.disabled = false;
                    progressContainer.style.display = "none";
                }
            })
            .catch(error => {
                finetuningStatus.textContent = "Error starting finetuning: " + error.message;
                runFinetuningBtn.disabled = false;
                progressContainer.style.display = "none";
            });
        });
    }
    
    function updateProgressBar() {
        if (!startTime) return;
        
        const currentTime = Date.now();
        const elapsedTime = currentTime - startTime;
        
        let progressPercent = Math.min(95, (elapsedTime / expectedDuration) * 100);
        
        progressBar.style.width = progressPercent + "%";
        progressText.textContent = Math.round(progressPercent) + "%";
        
        if (progressPercent < 95) {
            requestAnimationFrame(updateProgressBar);
        }
    }
    
    if (switchModelBtn) {
        switchModelBtn.addEventListener('click', function() {
            switchModelBtn.disabled = true;
            
            fetch('/switch_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    finetuningStatus.textContent = data.message;
                    currentModelSpan.textContent = data.model_name;
                } else {
                    finetuningStatus.textContent = "Error: " + data.message;
                }
                switchModelBtn.disabled = false;
            })
            .catch(error => {
                finetuningStatus.textContent = "Error switching model: " + error.message;
                switchModelBtn.disabled = false;
            });
        });
    }
    
    function checkFinetuningStatus() {
        fetch('/finetuning_status')
        .then(response => response.json())
        .then(data => {
            finetuningStatus.textContent = data.message;
            
            if (data.current_epoch && data.total_epochs) {
                const epochProgress = (data.current_epoch / data.total_epochs) * 100;
                progressBar.style.width = epochProgress + "%";
                progressText.textContent = Math.round(epochProgress) + "%";
            }
            
            if (!data.running) {
                clearInterval(statusCheckInterval);
                statusCheckInterval = null;
                
                runFinetuningBtn.disabled = false;
                
                progressBar.style.width = "100%";
                progressText.textContent = "100%";
                
                setTimeout(() => {
                    progressContainer.style.display = "none";
                }, 3000);
                
                updateFeedbackCount();
            }
        })
        .catch(error => {
            console.error("Error checking finetuning status:", error);
        });
    }
    
    function updateFeedbackCount() {
        fetch('/finetuning_status')
        .then(response => response.json())
        .then(data => {
            if (data.feedback_count) {
                feedbackCountSpan.textContent = data.feedback_count;
            }
        })
        .catch(error => {
            console.error("Error updating feedback count:", error);
        });
    }
});