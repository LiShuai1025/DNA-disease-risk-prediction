class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.uploadedFile = null;
        this.init();
    }

    async init() {
        try {
            updateStatus('System ready. Upload a dataset to begin analysis.');
            this.initTimelineChart();
            this.setupFileUpload();
            this.updateRankingDisplay();
            
        } catch (error) {
            console.error('Initialization failed:', error);
            updateStatus('Initialization failed: ' + error.message, true);
        }
    }

    // Setup file upload event listener
    setupFileUpload() {
        const fileUpload = document.getElementById('fileUpload');
        const fileUploadLabel = document.getElementById('fileUploadLabel');
        const fileInfo = document.getElementById('fileInfo');

        fileUpload.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                this.uploadedFile = file;
                fileInfo.textContent = `Selected: ${file.name} (${this.formatFileSize(file.size)})`;
                document.getElementById('loadDatasetBtn').disabled = false;
                updateStatus(`File "${file.name}" selected. Click "Process Data" to continue.`);
                
                // Preview the file
                this.previewFile(file);
            }
        });
    }

    // Format file size for display
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Preview uploaded file
    async previewFile(file) {
        const previewContainer = document.getElementById('dataPreview');
        
        if (file.name.endsWith('.csv')) {
            this.previewCSV(file, previewContainer);
        } else {
            previewContainer.innerHTML = `
                <div class="loading">
                    <p>Preview not available for ${file.name.split('.').pop().toUpperCase()} files</p>
                    <p>File will be processed after clicking "Process Data"</p>
                </div>
            `;
        }
    }

    // Preview CSV file
    previewCSV(file, container) {
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const csvData = e.target.result;
            const results = Papa.parse(csvData, {
                header: true,
                preview: 10, // Only preview first 10 rows
                skipEmptyLines: true
            });
            
            if (results.errors.length > 0) {
                container.innerHTML = `<p style="color: #e74c3c;">Error parsing CSV: ${results.errors[0].message}</p>`;
                return;
            }
            
            if (results.data.length === 0) {
                container.innerHTML = '<p>No data found in CSV file</p>';
                return;
            }
            
            // Create preview table
            let tableHTML = '<table>';
            
            // Header row
            tableHTML += '<tr>';
            Object.keys(results.data[0]).forEach(key => {
                tableHTML += `<th>${key}</th>`;
            });
            tableHTML += '</tr>';
            
            // Data rows
            results.data.forEach(row => {
                tableHTML += '<tr>';
                Object.values(row).forEach(value => {
                    tableHTML += `<td>${value}</td>`;
                });
                tableHTML += '</tr>';
            });
            
            tableHTML += '</table>';
            
            if (results.data.length >= 10) {
                tableHTML += `<p style="text-align: center; margin-top: 10px; color: #7f8c8d;">
                    Showing first 10 rows of ${results.data.length} total
                </p>`;
            }
            
            container.innerHTML = tableHTML;
        };
        
        reader.onerror = () => {
            container.innerHTML = '<p style="color: #e74c3c;">Error reading file</p>';
        };
        
        reader.readAsText(file);
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
                        min: 0
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
        if (!dataLoader.samples || dataLoader.samples.length === 0) return;
        
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
        
        // Update chart bounds
        this.timelineChart.options.scales.x.max = Math.max(50, dataLoader.samples.length);
        this.timelineChart.data.datasets[0].data = correctData;
        this.timelineChart.data.datasets[1].data = wrongData;
        this.timelineChart.update();
    }

    // Update ranking display
    updateRankingDisplay() {
        const rankingContainer = document.getElementById('rankingContainer');
        
        if (!dataLoader.isDataLoaded || !dataLoader.samples || dataLoader.samples.length === 0) {
            rankingContainer.innerHTML = `
                <div class="loading">
                    <p>Waiting for dataset...</p>
                </div>
            `;
            return;
        }

        const rankedSamples = dataLoader.getRankedSamples();
        
        if (rankedSamples.length === 0) {
            rankingContainer.innerHTML = '<p>No samples available for ranking</p>';
            return;
        }
        
        let html = '';
        
        rankedSamples.forEach((sample, index) => {
            const accuracy = sample.isCorrect !== null ? (sample.isCorrect ? 85 : 45) : 0;
            const riskClass = sample.actualRisk ? sample.actualRisk.toLowerCase() + '-risk' : '';
            
            html += `
                <div class="rank-item ${riskClass}">
                    <div>
                        <div class="sample-name">${sample.name || sample.id}</div>
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

    // Load and process dataset
    async loadDataset() {
        if (!this.uploadedFile) {
            alert('Please select a file first.');
            return;
        }

        try {
            updateStatus('Processing dataset...');
            showProgress(0);
            
            // Disable buttons during processing
            document.getElementById('loadDatasetBtn').disabled = true;
            document.getElementById('fileUpload').disabled = true;
            
            // Process the file
            await dataLoader.loadFromFile(this.uploadedFile, (progress) => {
                showProgress(progress);
                updateStatus(`Processing dataset... ${progress}%`);
            });
            
            this.isDataLoaded = true;
            
            // Enable train button
            document.getElementById('trainModelBtn').disabled = false;
            
            // Update preview with processed data info
            const fileInfo = document.getElementById('fileInfo');
            fileInfo.textContent = `${dataLoader.samples.length} samples loaded`;
            
            updateStatus(`Dataset processed successfully! ${dataLoader.samples.length} samples loaded. Click "Train Model" to continue.`);
            this.updateRankingDisplay();
            hideProgress();
            
        } catch (error) {
            console.error('Dataset processing failed:', error);
            updateStatus('Dataset processing failed: ' + error.message, true);
            document.getElementById('loadDatasetBtn').disabled = false;
            document.getElementById('fileUpload').disabled = false;
            hideProgress();
        }
    }

    // Train model (lightweight version)
    async trainModel() {
        if (!this.isDataLoaded) {
            alert('Please process dataset first.');
            return;
        }

        try {
            updateStatus('Preparing training data...');
            showProgress(0);
            
            // Disable buttons during training
            document.getElementById('trainModelBtn').disabled = true;
            document.getElementById('loadDatasetBtn').disabled = true;
            
            updateStatus('Starting model training...');
            
            // Use a simpler, faster model
            await diseaseModel.trainSimpleModel(
                dataLoader.samples,
                (progress) => {
                    showProgress(progress);
                    updateStatus(`Training model... ${progress}%`);
                }
            );
            
            this.isModelReady = true;
            
            // Enable prediction button
            document.getElementById('runPredictionBtn').disabled = false;
            
            updateStatus('Model training completed successfully!');
            this.updatePerformanceMetrics('Model trained successfully. Ready for predictions.');
            hideProgress();
            
        } catch (error) {
            console.error('Training failed:', error);
            updateStatus('Training failed: ' + error.message, true);
            document.getElementById('trainModelBtn').disabled = false;
            document.getElementById('loadDatasetBtn').disabled = false;
            hideProgress();
        }
    }

    // Run prediction (lightweight version)
    async runPrediction() {
        if (!this.isModelReady) {
            alert('Please train the model first.');
            return;
        }

        try {
            updateStatus('Running predictions...');
            showProgress(0);
            
            // Run predictions
            const results = await diseaseModel.predictSamples(
                dataLoader.samples,
                (progress) => {
                    showProgress(progress);
                    updateStatus(`Running predictions... ${progress}%`);
                }
            );
            
            // Update samples with predictions
            dataLoader.samples.forEach((sample, index) => {
                if (results[index]) {
                    sample.predictedRisk = results[index].predictedRisk;
                    sample.confidence = results[index].confidence;
                    sample.isCorrect = sample.actualRisk === sample.predictedRisk;
                }
            });
            
            // Update UI
            this.updateRankingDisplay();
            this.updateTimelineChart();
            
            // Calculate performance metrics
            const correctCount = dataLoader.samples.filter(s => s.isCorrect).length;
            const totalCount = dataLoader.samples.length;
            const accuracy = totalCount > 0 ? (correctCount / totalCount * 100).toFixed(1) : 0;
            
            this.updatePerformanceMetrics(`
                <strong>Prediction Results:</strong><br>
                • Accuracy: ${accuracy}% (${correctCount}/${totalCount} correct)<br>
                • Model: Lightweight Classifier<br>
                • Samples: ${totalCount} analyzed
            `);
            
            updateStatus(`Prediction completed! Accuracy: ${accuracy}%`);
            hideProgress();
            
        } catch (error) {
            console.error('Prediction failed:', error);
            updateStatus('Prediction failed: ' + error.message, true);
            hideProgress();
        }
    }

    // Update performance metrics display
    updatePerformanceMetrics(html) {
        const metricsContainer = document.getElementById('performanceMetrics');
        metricsContainer.innerHTML = html;
    }

    // Reset system
    resetSystem() {
        // Reset data
        if (dataLoader.samples) {
            dataLoader.samples.forEach(sample => {
                sample.predictedRisk = null;
                sample.confidence = null;
                sample.isCorrect = null;
            });
        }
        
        dataLoader.isDataLoaded = false;
        this.isModelReady = false;
        this.uploadedFile = null;
        
        // Reset UI
        document.getElementById('fileUpload').value = '';
        document.getElementById('fileUpload').disabled = false;
        document.getElementById('loadDatasetBtn').disabled = true;
        document.getElementById('trainModelBtn').disabled = true;
        document.getElementById('runPredictionBtn').disabled = true;
        
        document.getElementById('fileInfo').textContent = '';
        document.getElementById('dataPreview').innerHTML = `
            <div class="loading">
                <p>No dataset uploaded yet</p>
                <p style="font-size: 0.9em; margin-top: 10px;">
                    Supported formats: CSV, Excel<br>
                    Expected columns: Sequence, GC_Content, Disease_Risk, etc.
                </p>
            </div>
        `;
        
        this.updateRankingDisplay();
        this.updateTimelineChart();
        this.updatePerformanceMetrics('<p style="text-align: center; color: #7f8c8d;">Training and prediction metrics will appear here</p>');
        updateStatus('System reset. Upload a dataset to begin analysis.');
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
