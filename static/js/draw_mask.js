/*
 * Multi-lasso mask drawing logic.
 * - We overlay shapes on 'displayCanvas' (visible).
 * - We fill those shapes on 'maskCanvas' (hidden) to create a binary mask.
 * - "Undo" removes the last shape, "Reset" clears all shapes.
 * - "Submit Mask" sends the final mask + original image + chosen label to '/submit_mask'.
 */

// Grab references
const gradcamImg = document.getElementById("gradcamImg");
const displayCanvas = document.getElementById("displayCanvas");
const displayCtx = displayCanvas ? displayCanvas.getContext("2d") : null;
const maskCanvas = document.getElementById("maskCanvas");
const maskCtx = maskCanvas ? maskCanvas.getContext("2d") : null;

const toggleDrawBtn = document.getElementById("toggleDrawBtn");
const undoBtn = document.getElementById("undoBtn");
const resetBtn = document.getElementById("resetBtn");
const submitMaskBtn = document.getElementById("submitMaskBtn");

// We'll fetch the original image base64 from the hidden div if it exists
const originalImageElem = document.getElementById("originalImageData");
const originalImageBase64 = originalImageElem ? originalImageElem.textContent : "";

/** Radios for TB label */
function getTbLabel() {
    const radios = document.querySelectorAll('input[name="tb_label"]');
    for (let r of radios) {
        if (r.checked) {
            return r.value; // "TB" or "Normal"
        }
    }
    return "Unknown";
}

let shapes = [];
let currentShape = [];
let drawing = false;

// If there's a Grad-CAM image, set up the canvases after it loads.
if (gradcamImg) {
    gradcamImg.onload = () => {
        setupCanvas(gradcamImg.width, gradcamImg.height);
    };
    // If image is already cached
    if (gradcamImg.complete) {
        setupCanvas(gradcamImg.width, gradcamImg.height);
    }
}

/** Initialize the canvases to match the grad-cam image size */
function setupCanvas(width, height) {
    if (!displayCanvas || !maskCanvas) return;

    displayCanvas.width = width;
    displayCanvas.height = height;
    displayCanvas.style.width = width + "px";
    displayCanvas.style.height = height + "px";
    displayCanvas.style.display = "none"; // initially hidden

    maskCanvas.width = width;
    maskCanvas.height = height;

    // Fill the mask with black => no region selected
    maskCtx.fillStyle = "black";
    maskCtx.fillRect(0, 0, width, height);

    // Setup drawing context
    displayCtx.clearRect(0, 0, width, height);
    displayCtx.lineWidth = 2;
    displayCtx.strokeStyle = "red";
    displayCtx.lineCap = "round";

    // Reset shape data
    shapes = [];
    currentShape = [];
    drawing = false;

    // "Toggle Draw" is enabled, others are disabled until we draw
    toggleDrawBtn.disabled = false;
    undoBtn.disabled = true;
    resetBtn.disabled = true;
    submitMaskBtn.disabled = true;
}

// Toggle display of the drawing canvas
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

// Register mouse events for drawing
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
    // Draw all finalized shapes
    shapes.forEach(shape => drawPolyline(shape));
    // Draw in-progress shape
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

    // Enable shape-based buttons
    undoBtn.disabled = false;
    resetBtn.disabled = false;
    submitMaskBtn.disabled = false;

    redrawDisplay();
    currentShape = [];
}

/** Fill the shape on the hidden mask canvas in white (selected region) */
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

// Undo last shape
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

// Reset all shapes
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

// Rebuild the mask after undo
function rebuildMaskCanvas() {
    maskCtx.fillStyle = "black";
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    shapes.forEach(s => fillShapeOnMask(s));
}

// Submit final mask + original image to /submit_mask + the chosen TB label
if (submitMaskBtn) {
    submitMaskBtn.addEventListener("click", async () => {
        if (shapes.length === 0) {
            alert("No shapes drawn to submit.");
            return;
        }
        // 1) Grab final mask as base64
        const maskDataURL = maskCanvas.toDataURL("image/png");
        const base64Mask = maskDataURL.split(",")[1]; // remove data prefix

        // 2) originalImageBase64 is from the hidden div
        if (!originalImageBase64) {
            alert("No original image data found.");
            return;
        }

        // 3) Retrieve which radio is checked (TB or Normal)
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
    const statusBox = document.getElementById('finetuning-status');
    
    if (runFinetuningBtn) {
        runFinetuningBtn.addEventListener('click', async function() {
            if (!confirm('Start offline model refinement using collected feedback data? This may take several minutes.')) {
                return;
            }
            
            statusBox.innerHTML = '<p>Starting fine-tuning process...</p>';
            runFinetuningBtn.disabled = true;
            
            try {
                const response = await fetch('/run_finetuning', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    const result = await response.json();
                    statusBox.innerHTML = `
                        <p>Fine-tuning ${result.success ? 'completed!' : 'failed'}</p>
                        <p>${result.message}</p>
                    `;
                    
                    if (result.success) {
                        document.getElementById('switchModelBtn').style.display = 'inline-block';
                    }
                } else {
                    statusBox.innerHTML = '<p>Error: Failed to start fine-tuning process</p>';
                }
            } catch (error) {
                statusBox.innerHTML = `<p>Error: ${error.message}</p>`;
            } finally {
                runFinetuningBtn.disabled = false;
            }
        });
    }
    
    if (switchModelBtn) {
        switchModelBtn.addEventListener('click', async function() {
            try {
                const response = await fetch('/switch_model', {
                    method: 'POST'
                });
                
                if (response.ok) {
                    const result = await response.json();
                    document.getElementById('current-model').textContent = result.model_name;
                    statusBox.innerHTML = `<p>${result.message}</p>`;
                }
            } catch (error) {
                statusBox.innerHTML = `<p>Error switching model: ${error.message}</p>`;
            }
        });
    }
});