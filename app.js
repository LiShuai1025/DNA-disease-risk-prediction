class DNADiseaseDashboard {
    constructor() {
        this.isModelReady = false;
        this.isDataLoaded = false;
        this.timelineChart = null;
        this.uploadedFile = null;
        this.dataSource = null; // 'builtin' or 'uploaded'
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

    // ... 其他方法保持不变 ...

    // Train model (使用正确的方法名)
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
            document.getElementById('loadBuiltInBtn').disabled = true;
            
            updateStatus('Training advanced machine learning model...');
            
            // 修复：使用正确的方法名 trainModel 而不是 trainAdvancedModel
            await diseaseModel.trainModel(
                dataLoader.samples,
                (progress, message) => {
                    showProgress(progress);
                    updateStatus(`Training model... ${progress}% - ${message}`);
                }
            );
            
            this.isModelReady = true;
            
            // Enable prediction button
            document.getElementById('runPredictionBtn').disabled = false;
            
            updateStatus('Model training completed successfully! Click "Run Prediction" to analyze samples.');
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

    // Run prediction (使用正确的方法名)
    async runPrediction() {
        if (!this.isModelReady) {
            alert('Please train the model first.');
            return;
        }

        try {
            updateStatus('Running advanced predictions...');
            showProgress(0);
            
            // 修复：使用正确的方法名 predictSamples 而不是 predictSamplesAdvanced
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
            this.updatePerformanceMetrics();
            
            // Calculate performance metrics
            const correctCount = dataLoader.samples.filter(s => s.isCorrect).length;
            const totalPredicted = dataLoader.samples.filter(s => s.predictedRisk !== null).length;
            const accuracy = totalPredicted > 0 ? (correctCount / totalPredicted * 100).toFixed(1) : 0;
            
            updateStatus(`Prediction completed! Accuracy: ${accuracy}% (${correctCount}/${totalPredicted} correct)`);
            hideProgress();
            
        } catch (error) {
            console.error('Prediction failed:', error);
            updateStatus('Prediction failed: ' + error.message, true);
            hideProgress();
        }
    }

    // ... 其他方法保持不变 ...
}

// ... 其他代码保持不变 ...
