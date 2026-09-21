document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('paint-canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    const modelSelect = document.getElementById('model-select');
    const resultsBox = document.getElementById('prediction-results');

    // Canvas settings
    // The background is black via CSS, but we also fill it with black for the image export
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 25; // Thick stroke for better representation of digits
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function draw(e) {
        if (!isDrawing) return;
        
        // Handle mouse or touch events
        let clientX = e.clientX || (e.touches && e.touches[0].clientX);
        let clientY = e.clientY || (e.touches && e.touches[0].clientY);
        
        const rect = canvas.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();

        lastX = x;
        lastY = y;
    }

    function startDrawing(e) {
        isDrawing = true;
        let clientX = e.clientX || (e.touches && e.touches[0].clientX);
        let clientY = e.clientY || (e.touches && e.touches[0].clientY);
        const rect = canvas.getBoundingClientRect();
        lastX = clientX - rect.left;
        lastY = clientY - rect.top;
        e.preventDefault(); // Prevent scrolling on touch
    }

    function stopDrawing() {
        isDrawing = false;
    }

    // Event Listeners for Mouse
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Event Listeners for Touch
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);

    // Clear Canvas
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultsBox.innerHTML = '<p class="placeholder-text">Draw a character and click Predict to see the results.</p>';
    });

    // Predict
    predictBtn.addEventListener('click', async () => {
        // Get image data as base64 png
        const dataURL = canvas.toDataURL('image/png');
        const modelName = modelSelect.value;
        
        resultsBox.innerHTML = '<p class="placeholder-text">Predicting...</p>';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: dataURL,
                    model: modelName
                })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            if (data.error) {
                resultsBox.innerHTML = `<p class="placeholder-text" style="color: red;">Error: ${data.error}</p>`;
                return;
            }

            // Display results
            resultsBox.innerHTML = '';
            data.predictions.forEach((pred, index) => {
                const item = document.createElement('div');
                item.className = 'prediction-item';
                
                // Add star to first (highest confidence) prediction
                const prefix = index === 0 ? '⭐ ' : '';
                
                item.innerHTML = `
                    <span class="prediction-char">${prefix}'${pred.char}'</span>
                    <span class="prediction-prob">Confidence: ${pred.prob.toFixed(2)}%</span>
                `;
                resultsBox.appendChild(item);
            });
            
        } catch (error) {
            console.error('Error:', error);
            resultsBox.innerHTML = '<p class="placeholder-text" style="color: red;">Error communicating with server.</p>';
        }
    });
});
