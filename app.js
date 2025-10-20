class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.probabilityChart = null;
        this.uploadedFile = null;
        this.dataSource = null;
        this.trainingInProgress = false;
        this.predictionInProgress = false;
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
            updateStatus('Building optimized GRU model...');
            showProgress(0);
            
            // Disable all buttons during training
            document.getElementById('trainModelBtn').disabled = true;
            document.getElementById('loadDatasetBtn').disabled = true;
            document.getElementById('runPredictionBtn').disabled = true;
            
            // Show training info
            document.getElementById('dataSourceInfo').innerHTML = `
                <strong>Training Information:</strong> 
                Using efficient GRU model with ${Math.min(dataLoader.samples.length, 150)} balanced samples. 
                Early stopping enabled for optimal performance.
            `;
            document.getElementById('dataSourceInfo').style.display = 'block';
            
            updateStatus('Training efficient GRU model...');
            
            await diseaseModel.trainModel(
                dataLoader.samples,
                (progress, message) => {
                    showProgress(progress);
                    updateStatus(`Training model... ${progress}% - ${message}`);
                }
            );
            
            this.isModelReady = true;
            this.trainingInProgress = false;
            
            document.getElementById('runPredictionBtn').disabled = false;
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('loadDatasetBtn').disabled = false;
            
            const modelInfo = diseaseModel.getModelInfo();
            document.getElementById('modelType').textContent = `${modelInfo.type} - ${modelInfo.architecture}`;
            
            updateStatus('Model training completed! Ready for predictions.');
            this.updatePerformanceMetrics();
            hideProgress();
            
        } catch (error) {
            console.error('Training failed:', error);
            updateStatus('Training failed: ' + error.message, true);
            
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('loadDatasetBtn').disabled = false;
            this.trainingInProgress = false;
            hideProgress();
        }
    }

    async runPrediction() {
        if (!this.isModelReady) {
            alert('Please train the model first.');
            return;
        }

        if (this.predictionInProgress) {
            alert('Prediction is already in progress. Please wait.');
            return;
        }

        try {
            this.predictionInProgress = true;
            updateStatus('Running optimized predictions...');
            showProgress(0);
            
            // Disable buttons during prediction
            document.getElementById('runPredictionBtn').disabled = true;
            document.getElementById('trainModelBtn').disabled = true;
            
            const startTime = Date.now();
            
            const results = await diseaseModel.predictSamples(
                dataLoader.samples,
                (progress) => {
                    showProgress(progress);
                    updateStatus(`Running predictions... ${progress}%`);
                }
            );
            
            const endTime = Date.now();
            const predictionTime = ((endTime - startTime) / 1000).toFixed(2);
            
            // Update samples with predictions
            dataLoader.samples.forEach((sample, index) => {
                if (results[index]) {
                    sample.predictedRisk = results[index].predictedRisk;
                    sample.confidence = results[index].confidence;
                    sample.isCorrect = sample.actualRisk === sample.predictedRisk;
                    sample.probabilities = results[index].probabilities;
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
            
            updateStatus(`Prediction completed in ${predictionTime}s! Accuracy: ${accuracy}% (${correctCount}/${totalPredicted} correct)`);
            
            // Re-enable buttons
            document.getElementById('runPredictionBtn').disabled = false;
            document.getElementById('trainModelBtn').disabled = false;
            this.predictionInProgress = false;
            hideProgress();
            
        } catch (error) {
            console.error('Prediction failed:', error);
            updateStatus('Prediction failed: ' + error.message, true);
            
            // Re-enable buttons on error
            document.getElementById('runPredictionBtn').disabled = false;
            document.getElementById('trainModelBtn').disabled = false;
            this.predictionInProgress = false;
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
                        ${sample.probabilities ? `
                        <div class="probability-bars">
                            <div class="probability-bar">
                                <div class="probability-label">High Risk:</div>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill fill-high-risk" style="width: ${sample.probabilities.High * 100}%"></div>
                                </div>
                                <div class="probability-value">${(sample.probabilities.High * 100).toFixed(1)}%</div>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-label">Medium Risk:</div>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill fill-pathogenic" style="width: ${sample.probabilities.Medium * 100}%"></div>
                                </div>
                                <div class="probability-value">${(sample.probabilities.Medium * 100).toFixed(1)}%</div>
                            </div>
                            <div class="probability-bar">
                                <div class="probability-label">Low Risk:</div>
                                <div class="accuracy-bar">
                                    <div class="accuracy-fill fill-low-risk" style="width: ${sample.probabilities.Low * 100}%"></div>
                                </div>
                                <div class="probability-value">${(sample.probabilities.Low * 100).toFixed(1)}%</div>
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
        
        // Show feature explanation if available
        this.showFeatureExplanation(sample);
    }

    showFeatureExplanation(sample) {
        const features = sample.features || {};
        let explanation = '';
        
        if (features.gcContent !== undefined) {
            let gcLevel = 'Normal';
            if (features.gcContent > 60) gcLevel = 'High';
            else if (features.gcContent < 40) gcLevel = 'Low';
            
            explanation += `<li>GC含量: ${features.gcContent.toFixed(1)}% (${gcLevel})</li>`;
        }
        
        if (features.kmerFreq !== undefined) {
            let complexity = features.kmerFreq > 0.6 ? 'High Complexity' : 'Low Complexity';
            explanation += `<li>序列复杂度: ${(features.kmerFreq * 100).toFixed(1)}% (${complexity})</li>`;
        }
        
        if (features.entropy !== undefined) {
            let entropyLevel = features.entropy > 1.8 ? 'High Diversity' : 'Low Diversity';
            explanation += `<li>序列熵: ${features.entropy.toFixed(3)} (${entropyLevel})</li>`;
        }
        
        if (features.repeatScore !== undefined && features.repeatScore > 0) {
            explanation += `<li>重复模式得分: ${features.repeatScore.toFixed(1)} (Possible repeats detected)</li>`;
        }
        
        if (explanation) {
            const explanationDiv = document.createElement('div');
            explanationDiv.className = 'feature-explanation';
            explanationDiv.innerHTML = `
                <div class="sequence-title">特征分析</div>
                <ul>${explanation}</ul>
            `;
            
            const preview = document.getElementById('sequencePreview');
            preview.appendChild(explanationDiv);
        }
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
                labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                datasets: [
                    {
                        label: 'Average Probability',
                        data: [0, 0, 0],
                        backgroundColor: [
                            'rgba(231, 76, 60, 0.7)',
                            'rgba(243, 156, 18, 0.7)',
                            'rgba(39, 174, 96, 0.7)'
                        ],
                        borderColor: [
                            'rgba(231, 76, 60, 1)',
                            'rgba(243, 156, 18, 1)',
                            'rgba(39, 174, 96, 1)'
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
        
        const samplesWithPredictions = dataLoader.samples.filter(s => s.probabilities !== undefined);
        if (samplesWithPredictions.length === 0) return;
        
        const avgHighRisk = samplesWithPredictions.reduce((sum, s) => sum + s.probabilities.High, 0) / samplesWithPredictions.length;
        const avgMediumRisk = samplesWithPredictions.reduce((sum, s) => sum + s.probabilities.Medium, 0) / samplesWithPredictions.length;
        const avgLowRisk = samplesWithPredictions.reduce((sum, s) => sum + s.probabilities.Low, 0) / samplesWithPredictions.length;
        
        this.probabilityChart.data.datasets[0].data = [avgHighRisk, avgMediumRisk, avgLowRisk];
        this.probabilityChart.update();
    }

    updatePerformanceMetrics() {
        if (!this.isDataLoaded || !dataLoader.samples) return;
        
        const samples = dataLoader.samples;
        const predictedSamples = samples.filter(s => s.predictedRisk !== null);
        const correctPredictions = predictedSamples.filter(s => s.isCorrect).length;
        const totalPredicted = predictedSamples.length;
        const accuracy = totalPredicted > 0 ? (correctPredictions / totalPredicted * 100).toFixed(1) : 0;
        
        // Calculate class-wise accuracy
        const classStats = {High: {correct: 0, total: 0}, Medium: {correct: 0, total: 0}, Low: {correct: 0, total: 0}};
        
        predictedSamples.forEach(sample => {
            if (classStats[sample.actualRisk]) {
                classStats[sample.actualRisk].total++;
                if (sample.isCorrect) {
                    classStats[sample.actualRisk].correct++;
                }
            }
        });
        
        document.getElementById('accuracyValue').textContent = `${accuracy}%`;
        document.getElementById('samplesValue').textContent = samples.length;
        document.getElementById('correctValue').textContent = correctPredictions;
        document.getElementById('wrongValue').textContent = totalPredicted - correctPredictions;
        
        // Update model info
        if (this.isModelReady) {
            const modelInfo = diseaseModel.getModelInfo();
            document.getElementById('modelType').textContent = `${modelInfo.type}`;
        }
        
        // Show class-wise accuracy if available
        this.showClassAccuracy(classStats);
    }

    showClassAccuracy(classStats) {
        let accuracyInfo = '';
        
        Object.keys(classStats).forEach(riskClass => {
            const stats = classStats[riskClass];
            if (stats.total > 0) {
                const classAccuracy = (stats.correct / stats.total * 100).toFixed(1);
                accuracyInfo += `${riskClass}: ${classAccuracy}% (${stats.correct}/${stats.total})<br>`;
            }
        });
        
        if (accuracyInfo) {
            const existingInfo = document.getElementById('classAccuracyInfo');
            if (existingInfo) {
                existingInfo.innerHTML = `<strong>Class Accuracy:</strong><br>${accuracyInfo}`;
            } else {
                const infoDiv = document.createElement('div');
                infoDiv.id = 'classAccuracyInfo';
                infoDiv.className = 'file-info';
                infoDiv.style.marginTop = '10px';
                infoDiv.innerHTML = `<strong>Class Accuracy:</strong><br>${accuracyInfo}`;
                document.getElementById('rankingContainer').parentNode.insertBefore(infoDiv, document.getElementById('rankingContainer'));
            }
        }
    }

    resetSystem() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.uploadedFile = null;
        this.dataSource = null;
        this.trainingInProgress = false;
        this.predictionInProgress = false;
        
        dataLoader.samples = [];
        dataLoader.isDataLoaded = false;
        
        // Clear file input
        document.getElementById('fileUpload').value = '';
        document.getElementById('fileUploadLabel').innerHTML = '<span>📁</span> Upload CSV Dataset';
        document.getElementById('fileInfo').style.display = 'none';
        document.getElementById('dataSourceInfo').style.display = 'none';
        document.getElementById('loadDatasetBtn').disabled = true;
        document.getElementById('trainModelBtn').disabled = true;
        document.getElementById('runPredictionBtn').disabled = true;
        
        // Remove class accuracy info if exists
        const classAccuracyInfo = document.getElementById('classAccuracyInfo');
        if (classAccuracyInfo) {
            classAccuracyInfo.remove();
        }
        
        this.updateRankingDisplay();
        this.updatePerformanceMetrics();
        
        // Reset charts
        if (this.timelineChart) {
            this.timelineChart.data.datasets[0].data = [];
            this.timelineChart.data.datasets[1].data = [];
            this.timelineChart.update();
        }
        
        if (this.probabilityChart) {
            this.probabilityChart.data.datasets[0].data = [0, 0, 0];
            this.probabilityChart.update();
        }
        
        // Hide sequence preview
        document.getElementById('sequencePreview').style.display = 'none';
        
        updateStatus('System reset. Upload a CSV file with DNA sequences to begin analysis.');
        
        // Clear TensorFlow.js memory
        if (tf && tf.memory) {
            tf.disposeVariables();
        }
        
        // Clear prediction cache if exists
        if (diseaseModel.predictionCache) {
            diseaseModel.predictionCache.clear();
        }
    }
}

// Global functions for HTML buttons
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.style.color = isError ? '#e74c3c' : '#ecf0f1';
    }
}

function showProgress(percent) {
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    
    if (progressContainer && progressBar) {
        progressContainer.style.display = 'block';
        progressBar.style.width = percent + '%';
        progressBar.textContent = percent + '%';
    }
}

function hideProgress() {
    const progressContainer = document.getElementById('progressContainer');
    if (progressContainer) {
        progressContainer.style.display = 'none';
    }
}

function loadDataset() {
    if (window.dashboard) {
        window.dashboard.loadDataset();
    }
}

function trainModel() {
    if (window.dashboard) {
        window.dashboard.trainModel();
    }
}

function runPrediction() {
    if (window.dashboard) {
        window.dashboard.runPrediction();
    }
}

function resetSystem() {
    if (window.dashboard) {
        window.dashboard.resetSystem();
    }
}

// Initialize dashboard when page loads
window.addEventListener('load', () => {
    window.dashboard = new DNADiseaseDashboard();
});
