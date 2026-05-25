document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const cameraBtn = document.getElementById('camera-btn');
    const cameraContainer = document.getElementById('camera-container');
    const video = document.getElementById('video');
    const captureBtn = document.getElementById('capture-btn');
    const canvas = document.getElementById('canvas');
    const resultSection = document.getElementById('result-section');
    const previewImg = document.getElementById('preview-img');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceScore = document.getElementById('confidence-score');
    const resetBtn = document.getElementById('reset-btn');
    const loading = document.getElementById('loading');

    // Drag & Drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Camera
    cameraBtn.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            cameraContainer.classList.remove('hidden');
            cameraBtn.classList.add('hidden');
            dropZone.classList.add('hidden');
        } catch (err) {
            alert("Could not access camera: " + err);
        }
    });

    captureBtn.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        canvas.toBlob((blob) => {
            const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
            handleFile(file);
            
            // Stop stream
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            cameraContainer.classList.add('hidden');
        }, 'image/jpeg');
    });

    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        dropZone.classList.remove('hidden');
        cameraBtn.classList.remove('hidden');
        fileInput.value = '';
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropZone.classList.add('hidden');
            cameraBtn.classList.add('hidden');
            loading.classList.remove('hidden');
            
            uploadAndPredict(file);
        };
        reader.readAsDataURL(file);
    }

    async function uploadAndPredict(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            loading.classList.add('hidden');
            resultSection.classList.remove('hidden');

            if (data.error) {
                predictionLabel.textContent = "Error";
                confidenceScore.textContent = data.error;
                confidenceBar.style.width = '0%';
            } else {
                predictionLabel.textContent = data.label;
                const confidence = (data.confidence * 100).toFixed(1);
                confidenceScore.textContent = confidence + '%';
                setTimeout(() => {
                    confidenceBar.style.width = confidence + '%';
                }, 100);
            }

        } catch (error) {
            loading.classList.add('hidden');
            resultSection.classList.remove('hidden');
            predictionLabel.textContent = "Error";
            confidenceScore.textContent = "Connection failed";
        }
    }
});
