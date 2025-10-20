class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.init();
    }

    async init() {
        try {
            updateStatus('System initialized. Click "Load Dataset" to begin.');
            this.initTimelineChart();
            this.updateRankingDisplay();
            
        } catch (error) {
            console.error('Initialization failed:', error);
            updateStatus('Initialization failed: ' + error.message, true);
        }
    }

    // Initialize timeline chart
    initTimelineChart() {
        const ctx = document.getElementById('timelineChart').getContext('2d');
        
        this.timelineChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Correct Predictions',
                    data: [],
                    backgroundColor: '#27ae60',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }, {
                    label: 'Wrong Predictions',
                    data: [],
                    backgroundColor: '#e74c3c',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Sample Index'
                        },
                        min: 0,
                        max: 50
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Prediction Accuracy'
                        },
                        min: -0.5,
                        max: 1.5,
                        ticks: {
                            callback: function(value) {
                                return value === 1 ? 'Correct' : value === 0 ? 'Wrong' : '';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const point = context.raw;
                                return `Sample: ${point.sample.name}, Prediction: ${point.sample.predictedRisk}`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Update timeline chart
    updateTimelineChart() {
        const timelineData = dataLoader.getTimelineData();
        const correctData = [];
        const wrongData = [];
        
        timelineData.forEach(point => {
            if (point.y === 1) {
                correctData.push(point);
            } else {
                wrongData.push(point);
            }
        });
        
        this.timelineChart.data.datasets[0].data = correctData;
        this.timelineChart.data.datasets[1].data = wrongData;
        this.timelineChart.update();
    }

    // Update ranking display
    updateRankingDisplay() {
        const rankingContainer = document.getElementById('rankingContainer');
        
        if (!dataLoader.isDataLoaded) {
            rankingContainer.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <p>Waiting for dataset...</p>
                </div>
            `;
            return;
        }

        const rankedSamples = dataLoader.getRankedSamples();
        let html = '';
        
        rankedSamples.forEach((sample, index) => {
            const accuracy = sample.isCorrect !== null ? (sample.isCorrect ? 85 : 45) : 0;
            const riskClass = sample.actualRisk ? sample.actualRisk.toLowerCase() + '-risk' : '';
            
            html += `
                <div class="rank-item ${riskClass}">
                    <div>
                        <div class="sample-name">${sample.name}</div>
                        <div class="accuracy-bar">
                            <div class="accuracy-fill" style="width: ${accuracy}%"></div>
                        </div>
                    </div>
                    <div class="accuracy">${accuracy}%</div>
                </div>
            `;
        });
        
        rankingContainer.innerHTML = html;
    }

    // Load dataset
    async loadDataset() {
        try {
            updateStatus('Loading dataset...');
            showProgress(0);
            
            // Disable buttons during loading
            document.getElementById('loadDatasetBtn').disabled = true;
            
            // Simulate progressive loading
            for (let i = 0; i <= 100; i += 10) {
                await new Promise(resolve => setTimeout(resolve, 100));
                showProgress(i);
                updateStatus(`Loading dataset... ${i}%`);
            }
            
            // Load actual data
            await dataLoader.loadSampleData();
            this.isDataLoaded = true;
            
            // Enable train button
            document.getElementById('trainModelBtn').disabled = false;
            
            updateStatus('Dataset loaded successfully! Click "Train Model" to continue.');
            this.updateRankingDisplay();
            hideProgress();
            
        } catch (error) {
            console.error('Dataset loading failed:', error);
            updateStatus('Dataset loading failed: ' + error.message, true);
            document.getElementById('loadDatasetBtn').disabled = false;
            hideProgress();
        }
    }

    // Train model (optimized for performance)
    async trainModel() {
        if (!this.isDataLoaded) {
            alert('Please load dataset first.');
            return;
        }

        try {
            updateStatus('Preparing training data...');
            showProgress(0);
            
            // Disable buttons during training
            document.getElementById('trainModelBtn').disabled = true;
            document.getElementById('loadDatasetBtn').disabled = true;
            
            // Prepare training data
            dataLoader.prepareTrainingData();
            const trainingData = dataLoader.trainingData;
            
            updateStatus('Starting model training (this may take a moment)...');
            
            // Train with progress updates
            const progressCallback = (progress) => {
                showProgress(progress);
                updateStatus(`Training model... ${progress}%`);
            };
            
            await diseaseModel.trainModel(
                trainingData.sequences,
                trainingData.numericalFeatures,
                trainingData.labels,
                20, // Reduced epochs for performance
                progressCallback
            );
            
            this.isModelReady = true;
            
            // Enable prediction button
            document.getElementById('runPredictionBtn').disabled = false;
            
            updateStatus('Model training completed successfully!');
            hideProgress();
            
        } catch (error) {
            console.error('Training failed:', error);
            updateStatus('Training failed: ' + error.message, true);
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('loadDatasetBtn').disabled = false;
            hideProgress();
        }
    }

    // Run prediction
    async runPrediction() {
        if (!this.isModelReady) {
            alert('Please train the model first.');
            return;
        }

        try {
            updateStatus('Running predictions on test data...');
            showProgress(0);
            
            // Split data
            const testData = dataLoader.splitData(0.7);
            
            // Batch prediction with progress
            const predictions = await diseaseModel.predictBatch(
                testData.sequences,
                testData.numericalFeatures,
                (progress) => {
                    showProgress(progress);
                    updateStatus(`Running predictions... ${progress}%`);
                }
            );
            
            // Update sample data
            testData.sampleIndices.forEach((sampleIndex, i) => {
                const sample = dataLoader.samples[sampleIndex];
                const prediction = predictions[i];
                
                sample.predictedRisk = prediction.predictedRisk;
                sample.confidence = prediction.confidence;
                sample.isCorrect = sample.actualRisk === sample.predictedRisk;
            });
            
            // Update UI
            this.updateRankingDisplay();
            this.updateTimelineChart();
            
            // Calculate overall accuracy
            const testSamples = testData.sampleIndices.map(i => dataLoader.samples[i]);
            const correctCount = testSamples.filter(s => s.isCorrect).length;
            const accuracy = (correctCount / testSamples.length * 100).toFixed(1);
            
            updateStatus(`Prediction completed! Overall accuracy: ${accuracy}%`);
            hideProgress();
            
        } catch (error) {
            console.error('Prediction failed:', error);
            updateStatus('Prediction failed: ' + error.message, true);
            hideProgress();
        }
    }

    // Reset system
    resetSystem() {
        dataLoader.samples.forEach(sample => {
            sample.predictedRisk = null;
            sample.confidence = null;
            sample.isCorrect = null;
        });
        
        dataLoader.isDataLoaded = false;
        this.isModelReady = false;
        
        // Reset buttons
        document.getElementById('loadDatasetBtn').disabled = false;
        document.getElementById('trainModelBtn').disabled = true;
        document.getElementById('runPredictionBtn').disabled = true;
        
        this.updateRankingDisplay();
        this.updateTimelineChart();
        updateStatus('System reset. Click "Load Dataset" to start.');
    }
}

// Progress management
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

// Global functions
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.style.color = isError ? '#e74c3c' : '#ecf0f1';
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

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new DNADiseaseDashboard();
});
