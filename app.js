class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.probabilityChart = null;
        this.uploadedFile = null;
        this.dataSource = null;
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
            
            // Only disable the loadDatasetBtn, no longer referencing loadBuiltInBtn
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

        try {
            updateStatus('Building GRU neural network...');
            showProgress(0);
            
            // Only disable relevant buttons, no longer referencing loadBuiltInBtn
            document.getElementById('trainModelBtn').disabled = true;
            document.getElementById('loadDatasetBtn').disabled = true;
            
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

    updateRankingDisplay() {
        const container = document.getElementById('rankingContainer');
        
        if (!this.isDataLoaded || !dataLoader.samples || dataLoader.samples.length === 0) {
            container.innerHTML = `
                <div class="loading">
                    <div class="loading-spinner"></div>
                    <p>Upload a CSV file with DNA sequences to see predictions</p>
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
                <div class="${riskClass}" onclick="dashboard.showSequencePreview('${sample.id}')">
                    <div class="sample-info">
                        <div class="sample-name">${sample.name}</div>
                        <div class="sample-details">
                            Actual: ${sample.actualRisk} | Predicted: ${sample.predictedRisk || '--'} ${correctIcon}
                        </div>
                        ${sample.highRiskProbability !== undefined ? `
                        <div class="probability-bars">
                            <div class="probability-bar">
                                <div class="probability-label">High Risk:</div>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill fill-high-risk" style="width: ${sample.highRiskProbability * 100}%"></div>
                                </div>
                                <div class="probability-value">${(sample.highRiskProbability * 100).toFixed(1)}%</div>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-label">Pathogenic:</div>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill fill-pathogenic" style="width: ${sample.pathogenicProbability * 100}%"></div>
                                </div>
                                <div class="probability-value">${(sample.pathogenicProbability * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                        ` : ''}
                    </div>
                    <div class="sample-stats">
                        <div class="confidence">${confidence}</div>
                        ${sample.predictedRisk ? `<div class="risk-badge badge-${sample.predictedRisk.toLowerCase()}">${sample.predictedRisk}</div>` : ''}
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }

    showSequencePreview(sampleId) {
        const sample = dataLoader.samples.find(s => s.id === sampleId);
        if (!sample || !sample.sequence) return;
        
        const preview = document.getElementById('sequencePreview');
        const content = document.getElementById('sequenceContent');
        
        // Format sequence for better readability (groups of 10 bases)
        let formattedSequence = '';
        for (let i = 0; i < sample.sequence.length; i += 10) {
            formattedSequence += sample.sequence.substring(i, i + 10) + ' ';
            if ((i / 10 + 1) % 5 === 0) formattedSequence += '\n';
        }
        
        content.textContent = formattedSequence;
        preview.style.display = 'block';
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
                        fill: true,
                        data: []
                    },
                    {
                        label: 'Incorrect Predictions', 
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4,
                        fill: true,
                        data: []
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

    initProbabilityChart() {
        const ctx = document.getElementById('probabilityChart').getContext('2d');
        this.probabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['High Risk', 'Pathogenic'],
                datasets: [
                    {
                        label: 'Probability Distribution',
                        data: [0, 0],
                        backgroundColor: [
                            'rgba(231, 76, 60, 0.7)',
                            'rgba(243, 156, 18, 0.7)'
                        ],
                        borderColor: [
                            'rgba(231, 76, 60, 1)',
                            'rgba(243, 156, 18, 1)'
                        ],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {
                            display: true,
                            text: 'Probability'
                        }
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

    updateProbabilityChart() {
        if (!this.probabilityChart || !dataLoader.samples) return;
        
        const samplesWithPredictions = dataLoader.samples.filter(s => s.highRiskProbability !== undefined);
        if (samplesWithPredictions.length === 0) return;
        
        const avgHighRisk = samplesWithPredictions.reduce((sum, s) => sum + s.highRiskProbability, 0) / samplesWithPredictions.length;
        const avgPathogenic = samplesWithPredictions.reduce((sum, s) => sum + s.pathogenicProbability, 0) / samplesWithPredictions.length;
        
        this.probabilityChart.data.datasets[0].data = [avgHighRisk, avgPathogenic];
        this.probabilityChart.update();
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
        document.getElementById('fileUploadLabel').innerHTML = '<span>📁</span> Upload CSV Dataset';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('loadDatasetBtn').disabled = true;
        document.getElementById('trainModelBtn').disabled = true;
        document.getElementById('runPredictionBtn').disabled = true;
        
        this.updateRankingDisplay();
        this.updatePerformanceMetrics();
        this.updateTimelineChart();
        this.updateProbabilityChart();
        
        // Hide sequence preview
        document.getElementById('sequencePreview').style.display = 'none';
        
        updateStatus('System reset. Upload a CSV file with DNA sequences to begin analysis.');
        
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
