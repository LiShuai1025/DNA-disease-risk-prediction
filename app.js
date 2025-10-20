class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.uploadedFile = null;
        this.dataSource = null;
        this.init();
    }

    async init() {
        try {
            updateStatus('System ready. Use built-in data or upload your dataset.');
            this.initTimelineChart();
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

    async loadBuiltInDataset() {
        try {
            updateStatus('Loading built-in DNA dataset...');
            showProgress(0);
            
            document.getElementById('loadBuiltInBtn').disabled = true;
            document.getElementById('loadDatasetBtn').disabled = true;
            
            await dataLoader.loadBuiltInDataset((progress) => {
                showProgress(progress);
                updateStatus(`Loading built-in dataset... ${progress}%`);
            });
            
            this.isDataLoaded = true;
            this.dataSource = 'builtin';
            
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('fileInfo').textContent = `Built-in dataset: ${dataLoader.samples.length} DNA sequences`;
            document.getElementById('fileInfo').style.display = 'block';
            
            this.updateRankingDisplay();
            updateStatus(`Built-in dataset loaded successfully! ${dataLoader.samples.length} DNA sequences ready for training.`);
            hideProgress();
            
        } catch (error) {
            console.error('Failed to load built-in dataset:', error);
            updateStatus('Failed to load built-in dataset: ' + error.message, true);
            document.getElementById('loadBuiltInBtn').disabled = false;
            hideProgress();
        }
    }

    async loadDataset() {
        if (!this.uploadedFile) {
            alert('Please select a file first.');
            return;
        }

        try {
            updateStatus('Loading and processing DNA dataset...');
            showProgress(0);
            
            document.getElementById('loadDatasetBtn').disabled = true;
            document.getElementById('loadBuiltInBtn').disabled = true;
            
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
            document.getElementById('loadBuiltInBtn').disabled = false;
            hideProgress();
        }
    }

    async trainModel() {
        if (!this.isDataLoaded) {
            alert('Please load dataset first.');
            return;
        }

        try {
            updateStatus('Building GRU neural network...');
            showProgress(0);
            
            document.getElementById('trainModelBtn').disabled = true;
            document.getElementById('loadDatasetBtn').disabled = true;
            document.getElementById('loadBuiltInBtn').disabled = true;
            
            updateStatus('Training multi-output GRU model...');
            
            await diseaseModel.trainModel(
                dataLoader.samples,
                (progress, message) => {
                    showProgress(progress);
                    updateStatus(`Training GRU model... ${progress}% - ${message}`);
                }
            );
            
            this.isModelReady = true;
            document.getElementById('runPredictionBtn').disabled = false;
            
            const modelInfo = diseaseModel.getModelInfo();
            document.getElementById('modelType').textContent = `${modelInfo.type} - ${modelInfo.architecture}`;
            
            updateStatus('GRU model training completed! Click "Run Prediction" to analyze DNA sequences.');
            this.updatePerformanceMetrics();
            hideProgress();
            
        } catch (error) {
            console.error('Training failed:', error);
            updateStatus('Training failed: ' + error.message, true);
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('loadDatasetBtn').disabled = false;
            document.getElementById('loadBuiltInBtn').disabled = false;
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

    updateRankingDisplay() {
        const container = document.getElementById('rankingContainer');
        
        if (!this.isDataLoaded || !dataLoader.samples || dataLoader.samples.length === 0) {
            container.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <p>Use built-in data or upload a dataset to see rankings</p>
                </div>
            `;
            return;
        }

        const rankedSamples = dataLoader.getRankedSamples();
        const topSamples = rankedSamples.slice(0, 10);
        
        if (topSamples.length === 0) {
            container.innerHTML = `
                <div class="loading">
                    <p>No prediction data available. Run prediction to see rankings.</p>
                </div>
            `;
            return;
        }

        let html = '';
        topSamples.forEach((sample, index) => {
            const riskClass = sample.predictedRisk ? 
                `rank-item ${sample.predictedRisk.toLowerCase()}-risk` : 'rank-item';
            const confidence = sample.confidence ? (sample.confidence * 100).toFixed(1) + '%' : '--';
            const correctIcon = sample.isCorrect ? '✅' : '❌';
            
            html += `
                <div class="${riskClass}">
                    <div>
                        <div class="sample-name">${sample.name}</div>
                        <div style="font-size: 0.8em; color: #666;">
                            Actual: ${sample.actualRisk} | Predicted: ${sample.predictedRisk || '--'} ${correctIcon}
                        </div>
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: ${sample.confidence ? sample.confidence * 100 : 0}%"></div>
                        </div>
                    </div>
                    <div class="accuracy">${confidence}</div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    initTimelineChart() {
        const ctx = document.getElementById('timelineChart').getContext('2d');
        this.timelineChart = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: [
                    {
                        label: 'Correct Predictions',
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Incorrect Predictions', 
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Sample Sequence'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Prediction Accuracy'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    updateTimelineChart() {
        if (!this.timelineChart) return;
        
        const timelineData = dataLoader.getTimelineData();
        const correctData = [];
        const incorrectData = [];
        
        timelineData.forEach((point, index) => {
            if (point.y === 1) {
                correctData.push({x: index, y: 1});
                incorrectData.push({x: index, y: 0});
            } else {
                correctData.push({x: index, y: 0});
                incorrectData.push({x: index, y: 1});
            }
        });
        
        this.timelineChart.data.datasets[0].data = correctData;
        this.timelineChart.data.datasets[1].data = incorrectData;
        this.timelineChart.update();
    }

    updatePerformanceMetrics() {
        if (!this.isDataLoaded || !dataLoader.samples) return;
        
        const samples = dataLoader.samples;
        const predictedSamples = samples.filter(s => s.predictedRisk !== null);
        const correctPredictions = predictedSamples.filter(s => s.isCorrect).length;
        const totalPredicted = predictedSamples.length;
        const accuracy = totalPredicted > 0 ? (correctPredictions / totalPredicted * 100).toFixed(1) : 0;
        
        document.getElementById('accuracyValue').textContent = `${accuracy}%`;
        document.getElementById('samplesValue').textContent = samples.length;
        document.getElementById('correctValue').textContent = correctPredictions;
        document.getElementById('wrongValue').textContent = totalPredicted - correctPredictions;
        
        // Update model info
        if (this.isModelReady) {
            const modelInfo = diseaseModel.getModelInfo();
            document.getElementById('modelType').textContent = `${modelInfo.type}`;
        }
    }

    resetSystem() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.uploadedFile = null;
        this.dataSource = null;
        
        dataLoader.samples = [];
        dataLoader.isDataLoaded = false;
        
        document.getElementById('fileUpload').value = '';
        document.getElementById('fileUploadLabel').innerHTML = '<span>📁</span> Upload Dataset';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('loadDatasetBtn').disabled = true;
        document.getElementById('trainModelBtn').disabled = true;
        document.getElementById('runPredictionBtn').disabled = true;
        document.getElementById('loadBuiltInBtn').disabled = false;
        
        this.updateRankingDisplay();
        this.updatePerformanceMetrics();
        this.updateTimelineChart();
        
        updateStatus('System reset. Click "Use Built-in Data" or upload your dataset to begin.');
        
        // Clear TensorFlow.js memory
        if (tf && tf.memory) {
            tf.disposeVariables();
        }
    }
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

function loadBuiltInDataset() {
    window.dashboard.loadBuiltInDataset();
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
