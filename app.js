class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.init();
    }

    async init() {
        try {
            updateStatus('Loading sample data...');
            
            // Load data
            await dataLoader.loadSampleData();
            this.isDataLoaded = true;
            
            // Initialize chart
            this.initTimelineChart();
            
            // Update ranking display
            this.updateRankingDisplay();
            
            updateStatus('System ready. Click "Train Model" to start.');
            
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

    // Train model
    async trainModel() {
        if (!this.isDataLoaded) {
            alert('Please wait for data to load first.');
            return;
        }

        try {
            updateStatus('Preparing training data...');
            
            // Prepare training data
            dataLoader.prepareTrainingData();
            const trainingData = dataLoader.trainingData;
            
            updateStatus('Starting model training...');
            
            // Train model
            await diseaseModel.trainModel(
                trainingData.sequences,
                trainingData.numericalFeatures,
                trainingData.labels,
                30 // epochs
            );
            
            this.isModelReady = true;
            updateStatus('Model training completed successfully!');
            
        } catch (error) {
            console.error('Training failed:', error);
            updateStatus('Training failed: ' + error.message, true);
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
            
            // Split data
            const testData = dataLoader.splitData(0.7);
            
            // Batch prediction
            const predictions = await diseaseModel.predictBatch(
                testData.sequences,
                testData.numericalFeatures
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
            
        } catch (error) {
            console.error('Prediction failed:', error);
            updateStatus('Prediction failed: ' + error.message, true);
        }
    }

    // Reset system
    resetSystem() {
        dataLoader.samples.forEach(sample => {
            sample.predictedRisk = null;
            sample.confidence = null;
            sample.isCorrect = null;
        });
        
        this.updateRankingDisplay();
        this.updateTimelineChart();
        updateStatus('System reset. Ready for new training.');
    }
}

// Global functions
function updateStatus(message, isError = false) {
    const statusElement = document.getElementById('status');
    statusElement.textContent = message;
    statusElement.style.color = isError ? '#e74c3c' : '#ecf0f1';
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
