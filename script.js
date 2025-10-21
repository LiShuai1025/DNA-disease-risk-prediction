class DNAClassifier {
    constructor() {
        this.model = null;
        this.trainData = null;
        this.testData = null;
        this.isTraining = false;
        this.classLabels = ['Human', 'Bacteria', 'Virus', 'Plant'];
        this.featureNames = [
            'GC_Content', 'AT_Content', 'Sequence_Length', 
            'Num_A', 'Num_T', 'Num_C', 'Num_G', 'kmer_3_freq'
        ];
        this.modelType = 'improved_dense';
        this.trainingHistory = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.log('DNA Classifier System Initialized', 'success');
    }

    setupEventListeners() {
        // File upload listeners
        document.getElementById('trainFile').addEventListener('change', (e) => this.handleFileUpload(e, 'train'));
        document.getElementById('testFile').addEventListener('change', (e) => this.handleFileUpload(e, 'test'));
        document.getElementById('modelJsonFile').addEventListener('change', (e) => this.updateFileName(e, 'modelJson'));
        document.getElementById('modelWeightsFile').addEventListener('change', (e) => this.updateFileName(e, 'modelWeights'));

        // Model type selection
        document.getElementById('modelType').addEventListener('change', (e) => {
            this.setModelType(e.target.value);
        });

        // Button listeners
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModel());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.evaluateModel());
        document.getElementById('testRandomBtn').addEventListener('click', () => this.testRandomSamples());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.saveModel());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetSystem());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.loadModel());
        document.getElementById('testSingleBtn').addEventListener('click', () => this.testSingleSequence());
    }

    setModelType(type) {
        this.modelType = type;
        this.log(`Model architecture set to: ${type}`, 'success');
        document.getElementById('modelArchitecture').textContent = type;
    }

    log(message, type = 'info') {
        const logContainer = document.getElementById('logContainer');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // Also log to console for debugging
        console.log(`[${type.toUpperCase()}] ${message}`);
    }

    updateFileName(event, type) {
        const file = event.target.files[0];
        const fileNameElements = {
            'train': 'trainFileName',
            'test': 'testFileName',
            'modelJson': 'modelJsonFileName',
            'modelWeights': 'modelWeightsFileName'
        };
        
        if (file) {
            document.getElementById(fileNameElements[type]).textContent = file.name;
        }
    }

    async detectDelimiter(file) {
        return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target.result;
                const firstLine = content.split('\n')[0];
                
                // Detect delimiter
                if (firstLine.includes('\t')) {
                    resolve('\t');
                } else if (firstLine.includes(',')) {
                    resolve(',');
                } else if (firstLine.includes(';')) {
                    resolve(';');
                } else {
                    // Default to tab delimiter
                    resolve('\t');
                }
            };
            reader.readAsText(file);
        });
    }

    async handleFileUpload(event, dataType) {
        const file = event.target.files[0];
        if (!file) return;

        this.log(`Loading ${dataType === 'train' ? 'training' : 'testing'} data: ${file.name}`);
        
        try {
            // Auto-detect delimiter
            const delimiter = await this.detectDelimiter(file);
            this.log(`Detected delimiter: ${delimiter === '\t' ? 'tab' : delimiter}`);
            
            const data = await DataLoader.loadCSV(file, delimiter);
            
            if (dataType === 'train') {
                this.trainData = DataLoader.processData(data);
                document.getElementById('trainSamples').textContent = this.trainData.features.length;
                
                // Update class distribution
                const distribution = this.trainData.analysis.classDistribution;
                document.getElementById('classDistribution').textContent = JSON.stringify(distribution);
                
                this.log(`Training data loaded successfully: ${this.trainData.features.length} samples`);
                this.log(`Class distribution: ${JSON.stringify(distribution)}`);
                
                // Show data analysis
                if (this.trainData.analysis.featureStats.GC_Content) {
                    const stats = this.trainData.analysis.featureStats.GC_Content;
                    this.log(`GC Content analysis - Min: ${stats.min.toFixed(2)}, Max: ${stats.max.toFixed(2)}, Mean: ${stats.mean.toFixed(2)}`);
                }
            } else {
                this.testData = DataLoader.processData(data);
                document.getElementById('testSamples').textContent = this.testData.features.length;
                this.log(`Testing data loaded successfully: ${this.testData.features.length} samples`);
            }
            
            this.updateFileName(event, dataType);
        } catch (error) {
            this.log(`Error loading data: ${error.message}`, 'error');
            console.error('Detailed error:', error);
        }
    }

    async loadCSVWithDelimiter(file, delimiter) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                delimiter: delimiter,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error(results.errors[0].message));
                    } else {
                        resolve(results.data);
                    }
                },
                error: (error) => {
                    reject(error);
                }
            });
        });
    }

    async trainModel() {
        if (!this.trainData) {
            this.log('Error: Please upload training data first', 'error');
            return;
        }

        if (this.isTraining) {
            this.log('Training in progress, please wait...', 'warning');
            return;
        }

        // Show data statistics
        if (this.trainData.analysis) {
            this.log('Data Analysis Summary:');
            this.log(`Total samples: ${this.trainData.analysis.totalSamples}`);
            this.log(`Class distribution: ${JSON.stringify(this.trainData.analysis.classDistribution)}`);
        }

        this.isTraining = true;
        this.log(`Starting model training with ${this.modelType} architecture...`);

        let xs, ys;
        try {
            // Create model
            this.model = ModelBuilder.createModel(
                this.trainData.features[0].length, 
                this.classLabels.length,
                this.modelType
            );
            
            const { features, labels } = this.trainData;
            xs = tf.tensor2d(features);
            ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

            // Training configuration
            const epochs = 100;
            const batchSize = Math.min(32, features.length);
            const validationSplit = 0.2;

            let bestValAcc = 0;
            let patience = 10;
            let patienceCounter = 0;

            this.trainingHistory = [];
            
            await this.model.fit(xs, ys, {
                epochs: epochs,
                batchSize: batchSize,
                validationSplit: validationSplit,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        const currentValAcc = logs.val_acc;
                        const currentAcc = logs.acc;
                        const currentLoss = logs.loss;
                        const currentValLoss = logs.val_loss;
                        
                        // Store history for visualization
                        this.trainingHistory.push({
                            epoch: epoch + 1,
                            accuracy: currentAcc,
                            val_accuracy: currentValAcc,
                            loss: currentLoss,
                            val_loss: currentValLoss
                        });
                        
                        this.log(`Epoch ${epoch + 1}/${epochs} - ` +
                                `Accuracy: ${(currentAcc * 100).toFixed(2)}%, ` +
                                `Val Accuracy: ${(currentValAcc * 100).toFixed(2)}%, ` +
                                `Loss: ${currentLoss.toFixed(4)}, ` +
                                `Val Loss: ${currentValLoss.toFixed(4)}`);

                        // Early stopping
                        if (currentValAcc > bestValAcc) {
                            bestValAcc = currentValAcc;
                            patienceCounter = 0;
                            this.log(`New best validation accuracy: ${(bestValAcc * 100).toFixed(2)}%`, 'success');
                        } else {
                            patienceCounter++;
                        }

                        if (patienceCounter >= patience) {
                            this.log(`Early stopping triggered at epoch ${epoch + 1}`, 'warning');
                            this.model.stopTraining = true;
                        }
                    },
                    onTrainEnd: () => {
                        this.log(`Training completed. Best validation accuracy: ${(bestValAcc * 100).toFixed(2)}%`, 'success');
                        // Draw training history if we have data
                        Visualization.drawTrainingHistory(this.trainingHistory);
                    }
                }
            });

            this.log('Model training completed successfully!', 'success');
            this.updateModelInfo();

            // Evaluate on training data
            await this.evaluateOnTrainData();

        } catch (error) {
            this.log(`Training error: ${error.message}`, 'error');
            console.error('Training error details:', error);
        } finally {
            this.isTraining = false;
            
            // Clean up memory
            if (xs) xs.dispose();
            if (ys) ys.dispose();
        }
    }

    async evaluateOnTrainData() {
        if (!this.model || !this.trainData) return;

        this.log('Evaluating model on training data...');
        
        let xs, ys, evaluation;
        try {
            const { features, labels } = this.trainData;
            xs = tf.tensor2d(features);
            ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

            evaluation = this.model.evaluate(xs, ys);
            const loss = evaluation[0].dataSync()[0];
            const accuracy = evaluation[1].dataSync()[0];

            this.log(`Training set evaluation - Accuracy: ${(accuracy * 100).toFixed(2)}%, Loss: ${loss.toFixed(4)}`, 
                     accuracy > 0.7 ? 'success' : 'warning');
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`, 'error');
        } finally {
            if (xs) xs.dispose();
            if (ys) ys.dispose();
            if (evaluation) {
                evaluation.forEach(tensor => tensor.dispose());
            }
        }
    }

    async evaluateModel() {
        if (!this.model) {
            this.log('Error: No model available. Please train or load a model first.', 'error');
            return;
        }

        if (!this.testData) {
            this.log('Error: Please upload test data first.', 'error');
            return;
        }

        this.log('Starting model evaluation on test data...');

        let xs, ys, evaluation;
        try {
            const { features, labels } = this.testData;
            xs = tf.tensor2d(features);
            ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

            evaluation = this.model.evaluate(xs, ys);
            const loss = evaluation[0].dataSync()[0];
            const accuracy = evaluation[1].dataSync()[0];

            const resultType = accuracy > 0.7 ? 'success' : accuracy > 0.5 ? 'warning' : 'error';
            this.log(`Test set evaluation - Accuracy: ${(accuracy * 100).toFixed(2)}%, Loss: ${loss.toFixed(4)}`, resultType);
        } catch (error) {
            this.log(`Evaluation error: ${error.message}`, 'error');
        } finally {
            if (xs) xs.dispose();
            if (ys) ys.dispose();
            if (evaluation) {
                evaluation.forEach(tensor => tensor.dispose());
            }
        }
    }

    async testRandomSamples() {
        if (!this.model || !this.testData) {
            this.log('Error: No model or test data available.', 'error');
            return;
        }

        const { features, labels, rawData } = this.testData;
        const resultsContainer = document.getElementById('randomTestResults');
        resultsContainer.innerHTML = '';

        // Randomly select 5 samples
        const indices = [];
        const sampleCount = Math.min(5, features.length);
        for (let i = 0; i < sampleCount; i++) {
            indices.push(Math.floor(Math.random() * features.length));
        }

        let correctPredictions = 0;

        for (const index of indices) {
            const feature = features[index];
            const trueLabel = labels[index];
            const sequence = rawData[index].Sequence || 'N/A';

            let inputTensor, prediction;
            try {
                inputTensor = tf.tensor2d([feature]);
                prediction = this.model.predict(inputTensor);
                const results = await prediction.data();
                
                const predictedClassIndex = results.indexOf(Math.max(...results));
                const predictedClass = this.classLabels[predictedClassIndex];
                const trueClass = this.classLabels[trueLabel];
                const confidence = Math.max(...results) * 100;

                if (predictedClass === trueClass) {
                    correctPredictions++;
                }

                const resultDiv = document.createElement('div');
                resultDiv.className = `random-test-item ${predictedClass === trueClass ? '' : 'error'}`;
                resultDiv.innerHTML = `
                    <strong>Sample ${index + 1}</strong><br>
                    <small>Sequence: ${sequence.substring(0, 50)}${sequence.length > 50 ? '...' : ''}</small><br>
                    True Label: <strong>${trueClass}</strong> | Predicted: <strong>${predictedClass}</strong><br>
                    Confidence: ${confidence.toFixed(2)}%<br>
                    ${predictedClass === trueClass ? '✅ Correct' : '❌ Incorrect'}
                `;

                resultsContainer.appendChild(resultDiv);
            } finally {
                if (inputTensor) inputTensor.dispose();
                if (prediction) prediction.dispose();
            }
        }

        const accuracy = (correctPredictions / sampleCount) * 100;
        this.log(`Random testing completed. Accuracy: ${accuracy.toFixed(2)}% (${correctPredictions}/${sampleCount} correct)`, 
                 accuracy > 70 ? 'success' : 'warning');
    }

    async testSingleSequence() {
        const sequenceInput = document.getElementById('singleSequence').value.trim().toUpperCase();
        
        if (!sequenceInput) {
            this.log('Error: Please enter a DNA sequence.', 'error');
            return;
        }

        if (!/^[ATCG]+$/.test(sequenceInput)) {
            this.log('Error: DNA sequence can only contain A, T, C, G characters.', 'error');
            return;
        }

        if (!this.model) {
            this.log('Error: No model available. Please train or load a model first.', 'error');
            return;
        }

        let inputTensor, prediction;
        try {
            // Extract features from sequence
            const features = DataLoader.extractFeaturesFromSequence(sequenceInput);
            inputTensor = tf.tensor2d([features]);
            prediction = this.model.predict(inputTensor);
            const results = await prediction.data();
            
            const maxConfidence = Math.max(...results);
            const predictedClassIndex = results.indexOf(maxConfidence);
            const predictedClass = this.classLabels[predictedClassIndex];
            
            // Draw confidence chart
            Visualization.drawConfidenceChart(Array.from(results), this.classLabels);
            
            const resultDiv = document.getElementById('singleTestResult');
            resultDiv.innerHTML = `
                <div class="prediction-result">Prediction: ${predictedClass}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${maxConfidence * 100}%"></div>
                </div>
                <div>Confidence: ${(maxConfidence * 100).toFixed(2)}%</div>
                <div class="confidence-breakdown">
                    ${this.classLabels.map((label, index) => 
                        `${label}: ${(results[index] * 100).toFixed(2)}%`
                    ).join(' | ')}
                </div>
                <div class="feature-analysis">
                    <h4>Sequence Analysis</h4>
                    <div class="feature-grid">
                        <div class="feature-item">
                            <span class="feature-name">GC Content:</span>
                            <span class="feature-value">${(features[0] * 100).toFixed(2)}%</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">AT Content:</span>
                            <span class="feature-value">${(features[1] * 100).toFixed(2)}%</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">Sequence Length:</span>
                            <span class="feature-value">${features[2]}</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">A Count:</span>
                            <span class="feature-value">${features[3]}</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">T Count:</span>
                            <span class="feature-value">${features[4]}</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">C Count:</span>
                            <span class="feature-value">${features[5]}</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">G Count:</span>
                            <span class="feature-value">${features[6]}</span>
                        </div>
                        <div class="feature-item">
                            <span class="feature-name">3-mer Frequency:</span>
                            <span class="feature-value">${features[7].toFixed(4)}</span>
                        </div>
                    </div>
                </div>
            `;

            this.log(`Single sequence test completed: ${predictedClass} (${(maxConfidence * 100).toFixed(2)}% confidence)`, 'success');
            
        } catch (error) {
            this.log(`Single sequence test error: ${error.message}`, 'error');
            console.error('Test error details:', error);
        } finally {
            if (inputTensor) inputTensor.dispose();
            if (prediction) prediction.dispose();
        }
    }

    async saveModel() {
        if (!this.model) {
            this.log('Error: No model to save.', 'error');
            return;
        }

        this.log('Saving model...');

        try {
            // Save model architecture
            const modelJson = this.model.toJSON();
            const modelJsonStr = JSON.stringify(modelJson);
            const modelJsonBlob = new Blob([modelJsonStr], { type: 'application/json' });
            
            // Trigger download
            const modelJsonUrl = URL.createObjectURL(modelJsonBlob);
            const modelJsonLink = document.createElement('a');
            modelJsonLink.href = modelJsonUrl;
            modelJsonLink.download = 'dna-classifier-model.json';
            modelJsonLink.click();

            // Save weights
            await this.model.save('downloads://dna-classifier-model');

            this.log('Model saved successfully! Check your downloads folder for dna-classifier-model.json and dna-classifier-model.weights.bin', 'success');
        } catch (error) {
            this.log(`Model save error: ${error.message}`, 'error');
            console.error('Save error details:', error);
        }
    }

    async loadModel() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];

        if (!jsonFile || !weightsFile) {
            this.log('Error: Please select both model JSON and weights files.', 'error');
            return;
        }

        this.log('Loading model...');

        try {
            const modelJson = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    try {
                        resolve(JSON.parse(reader.result));
                    } catch (parseError) {
                        reject(new Error('Invalid JSON file'));
                    }
                };
                reader.onerror = () => reject(new Error('Failed to read JSON file'));
                reader.readAsText(jsonFile);
            });

            const modelWeights = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(new Error('Failed to read weights file'));
                reader.readAsArrayBuffer(weightsFile);
            });

            this.model = await tf.loadLayersModel(tf.io.fromMemory(modelJson, modelWeights));
            this.log('Model loaded successfully!', 'success');
            this.updateModelInfo();
        } catch (error) {
            this.log(`Model load error: ${error.message}`, 'error');
            console.error('Load error details:', error);
        }
    }

    updateModelInfo() {
        if (!this.model) return;

        let totalParams = 0;
        let layersCount = 0;

        this.model.summary(null, null, (line) => {
            if (line.includes('_________________________________________________________________')) return;
            
            const layerMatch = line.match(/^(\w+)\s+\((\w+)\)/);
            if (layerMatch) {
                layersCount++;
            }
            
            const paramMatch = line.match(/params:\s+([\d,]+)/);
            if (paramMatch) {
                const params = parseInt(paramMatch[1].replace(/,/g, ''));
                totalParams += params;
            }
        });

        document.getElementById('layersCount').textContent = layersCount;
        document.getElementById('totalParams').textContent = totalParams.toLocaleString();
        document.getElementById('modelArchitecture').textContent = this.modelType;

        this.log(`Model information updated: ${layersCount} layers, ${totalParams.toLocaleString()} parameters`);
    }

    resetSystem() {
        // Clear TensorFlow.js memory
        if (this.model) {
            this.model.dispose();
        }
        tf.disposeVariables();

        // Clear visualization charts
        Visualization.clearCharts();

        // Reset data and state
        this.model = null;
        this.trainData = null;
        this.testData = null;
        this.trainingHistory = [];
        
        // Reset UI elements
        document.getElementById('trainSamples').textContent = '0';
        document.getElementById('testSamples').textContent = '0';
        document.getElementById('layersCount').textContent = '0';
        document.getElementById('totalParams').textContent = '0';
        document.getElementById('classDistribution').textContent = '-';
        document.getElementById('modelArchitecture').textContent = '-';
        document.getElementById('logContainer').innerHTML = '';
        document.getElementById('randomTestResults').innerHTML = '';
        document.getElementById('singleTestResult').innerHTML = '';
        
        // Reset file inputs
        document.getElementById('trainFile').value = '';
        document.getElementById('testFile').value = '';
        document.getElementById('modelJsonFile').value = '';
        document.getElementById('modelWeightsFile').value = '';
        document.getElementById('singleSequence').value = '';
        
        document.getElementById('trainFileName').textContent = 'No file chosen';
        document.getElementById('testFileName').textContent = 'No file chosen';
        document.getElementById('modelJsonFileName').textContent = 'No file chosen';
        document.getElementById('modelWeightsFileName').textContent = 'No file chosen';

        // Reset model type
        document.getElementById('modelType').value = 'improved_dense';
        this.modelType = 'improved_dense';

        this.log('System reset completed. All memory cleared.', 'success');
    }
}

// Initialize application
let dnaClassifier;
document.addEventListener('DOMContentLoaded', () => {
    try {
        dnaClassifier = new DNAClassifier();
        console.log('DNA Classifier application initialized successfully');
    } catch (error) {
        console.error('Failed to initialize DNA Classifier:', error);
        alert('Error initializing application. Please check the console for details.');
    }
});
