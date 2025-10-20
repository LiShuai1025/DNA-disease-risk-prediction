class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.probabilityChart = null;
        this.uploadedFile = null;
        this.dataSource = null;
        this.trainingInProgress = false;
        this.init();
    }

    async init() {
        try {
            updateStatus('Upload a CSV file with DNA sequences to begin analysis');
            this.initTimelineChart();
            this.initProbabilityChart();
            this.setupFileUpload();
            this.updateRankingDisplay();
            this.updatePerformanceMetrics();
            
        } catch (error) {
            console.error('Initialization failed:', error);
            updateStatus('Initialization failed: ' + error.message, true);
        }
    }

    setupFileUpload() {
        const fileUpload = document.getElementById('fileUpload');
        const fileUploadLabel = document.getElementById('fileUploadLabel');
        
        fileUpload.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                this.uploadedFile = file;
                fileUploadLabel.innerHTML = `<span>📁</span> ${file.name}`;
                document.getElementById('loadDatasetBtn').disabled = false;
                updateStatus(`File selected: ${file.name}. Click "Process Data" to load.`);
            }
        });
    }

    async loadDataset() {
        if (!this.uploadedFile) {
            alert('Please select a CSV file first.');
            return;
        }

        try {
            updateStatus('Loading and processing DNA dataset...');
            showProgress(0);
            
            document.getElementById('loadDatasetBtn').disabled = true;
            
            await dataLoader.loadFromFile(this.uploadedFile, (progress) => {
                showProgress(progress);
                updateStatus(`Processing DNA data... ${progress}%`);
            });
            
            this.isDataLoaded = true;
            this.dataSource = 'uploaded';
            
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('fileInfo').textContent = `Uploaded file: ${this.uploadedFile.name} (${dataLoader.samples.length} sequences)`;
            document.getElementById('fileInfo').style.display = 'block';
            
            this.updateRankingDisplay();
            updateStatus(`Dataset loaded successfully! ${dataLoader.samples.length} DNA sequences ready for training.`);
            hideProgress();
            
        } catch (error) {
            console.error('Failed to load dataset:', error);
            updateStatus('Failed to load dataset: ' + error.message, true);
            document.getElementById('loadDatasetBtn').disabled = false;
            hideProgress();
        }
    }

    async trainModel() {
        if (!this.isDataLoaded) {
            alert('Please load dataset first.');
            return;
        }

        if (this.trainingInProgress) {
            alert('Training is already in progress. Please wait.');
            return;
        }

        try {
            this.trainingInProgress = true;
            updateStatus('Building optimized GRU neural network...');
            showProgress(0);
            
            // Disable all buttons during training
            document.getElementById('trainModelBtn').disabled = true;
            document.getElementById('loadDatasetBtn').disabled = true;
            document.getElementById('runPredictionBtn').disabled = true;
            
            // Show training info
            document.getElementById('dataSourceInfo').innerHTML = `
                <strong>Training Information:</strong> 
                Using optimized GRU model with ${Math.min(dataLoader.samples.length, 200)} samples. 
                This may take 1-3 minutes depending on your device.
            `;
            document.getElementById('dataSourceInfo').style.display = 'block';
            
            updateStatus('Training optimized GRU model (this may take a while)...');
            
            await diseaseModel.trainModel(
                dataLoader.samples,
                (progress, message) => {
                    showProgress(progress);
                    updateStatus(`Training GRU model... ${progress}% - ${message}`);
                }
            );
            
            this.isModelReady = true;
            this.trainingInProgress = false;
            
            document.getElementById('runPredictionBtn').disabled = false;
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('loadDatasetBtn').disabled = false;
            
            const modelInfo = diseaseModel.getModelInfo();
            document.getElementById('modelType').textContent = `${modelInfo.type} - ${modelInfo.architecture}`;
            
            updateStatus('GRU model training completed! Click "Run Prediction" to analyze DNA sequences.');
            this.updatePerformanceMetrics();
            hideProgress();
            
        } catch (error) {
            console.error('Training failed:', error);
            updateStatus('Training failed: ' + error.message, true);
            
            // Re-enable buttons on error
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('loadDatasetBtn').disabled = false;
            this.trainingInProgress = false;
            hideProgress();
        }
    }

    async runPrediction() {
        if (!this.isModelReady) {
            alert('Please train the GRU model first.');
            return;
        }

        try {
            updateStatus('Running GRU model predictions...');
            showProgress(0);
            
            const results = await diseaseModel.predictSamples(
                dataLoader.samples,
                (progress) => {
                    showProgress(progress);
                    updateStatus(`Running GRU predictions... ${progress}%`);
                }
            );
            
            // Update samples with predictions
            dataLoader.samples.forEach((sample, index) => {
                if (results[index]) {
                    sample.predictedRisk = results[index].predictedRisk;
                    sample.confidence = results[index].confidence;
                    sample.isCorrect = sample.actualRisk === sample.predictedRisk;
                    sample.highRiskProbability = results[index].highRiskProbability;
                    sample.pathogenicProbability = results[index].pathogenicProbability;
                }
            });
            
            // Update UI
            this.updateRankingDisplay();
            this.updateTimelineChart();
            this.updateProbabilityChart();
            this.updatePerformanceMetrics();
            
            // Calculate performance metrics
            const correctCount = dataLoader.samples.filter(s => s.isCorrect).length;
            const totalPredicted = dataLoader.samples.filter(s => s.predictedRisk !== null).length;
            const accuracy = totalPredicted > 0 ? (correctCount / totalPredicted * 100).toFixed(1) : 0;
            
            updateStatus(`GRU prediction completed! Accuracy: ${accuracy}% (${correctCount}/${totalPredicted} correct)`);
            hideProgress();
            
        } catch (error) {
            console.error('Prediction failed:', error);
            updateStatus('Prediction failed: ' + error.message, true);
            hideProgress();
        }
    }

    // ... rest of the methods remain the same as previous version ...
}

// Global functions for HTML buttons
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.style.color = isError ? '#e74c3c' : '#ecf0f1';
}

function showProgress(percent) {
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    
    progressContainer.style.display = 'block';
    progressBar.style.width = percent + '%';
    progressBar.textContent = percent + '%';
}

function hideProgress() {
    const progressContainer = document.getElementById('progressContainer');
    progressContainer.style.display = 'none';
}

function loadDataset() {
    window.dashboard.loadDataset();
}

function trainModel() {
    window.dashboard.trainModel();
}

function runPrediction() {
    window.dashboard.runPrediction();
}

function resetSystem() {
    window.dashboard.resetSystem();
}

// Initialize dashboard when page loads
window.addEventListener('load', () => {
    window.dashboard = new DNADiseaseDashboard();
});
