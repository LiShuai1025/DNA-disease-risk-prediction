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
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.log('System initialized');
    }

    setupEventListeners() {
        // File upload listeners
        document.getElementById('trainFile').addEventListener('change', (e) => this.handleFileUpload(e, 'train'));
        document.getElementById('testFile').addEventListener('change', (e) => this.handleFileUpload(e, 'test'));
        document.getElementById('modelJsonFile').addEventListener('change', (e) => this.updateFileName(e, 'modelJson'));
        document.getElementById('modelWeightsFile').addEventListener('change', (e) => this.updateFileName(e, 'modelWeights'));

        // Button listeners
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModel());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.evaluateModel());
        document.getElementById('testRandomBtn').addEventListener('click', () => this.testRandomSamples());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.saveModel());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetSystem());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.loadModel());
        document.getElementById('testSingleBtn').addEventListener('click', () => this.testSingleSequence());
    }

    log(message) {
        const logContainer = document.getElementById('logContainer');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
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

    async handleFileUpload(event, dataType) {
        const file = event.target.files[0];
        if (!file) return;

        this.log(`Loading ${dataType === 'train' ? 'training' : 'testing'} data: ${file.name}`);
        
        try {
            const data = await DataLoader.loadCSV(file);
            
            if (dataType === 'train') {
                this.trainData = DataLoader.processData(data);
                document.getElementById('trainSamples').textContent = this.trainData.features.length;
                this.log(`Training data loaded: ${this.trainData.features.length} samples`);
            } else {
                this.testData = DataLoader.processData(data);
                document.getElementById('testSamples').textContent = this.testData.features.length;
                this.log(`Testing data loaded: ${this.testData.features.length} samples`);
            }
            
            this.updateFileName(event, dataType);
        } catch (error) {
            this.log(`Error: Failed to load data - ${error.message}`);
        }
    }

    async trainModel() {
        if (!this.trainData) {
            this.log('Error: Please upload training data first');
            return;
        }

        if (this.isTraining) {
            this.log('Training in progress, please wait...');
            return;
        }

        this.isTraining = true;
        this.log('Starting model training...');

        try {
            // Create or reset model
            this.model = ModelBuilder.createModel(this.trainData.features[0].length, this.classLabels.length);
            
            const { features, labels } = this.trainData;
            const xs = tf.tensor2d(features);
            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

            // Training configuration
            const epochs = 50;
            const batchSize = 32;
            const validationSplit = 0.2;

            await this.model.fit(xs, ys, {
                epochs: epochs,
                batchSize: batchSize,
                validationSplit: validationSplit,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.log(`Epoch ${epoch + 1}/${epochs} - Accuracy: ${(logs.acc * 100).toFixed(2)}%, Val Accuracy: ${(logs.val_acc * 100).toFixed(2)}%, Loss: ${logs.loss.toFixed(4)}`);
                    }
                }
            });

            this.log('Training completed!');
            this.updateModelInfo();

        } catch (error) {
            this.log(`Training error: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    async evaluateModel() {
        if (!this.model) {
            this.log('Error: No model available, please train or load a model first');
            return;
        }

        if (!this.testData) {
            this.log('Error: Please upload test data first');
            return;
        }

        this.log('Starting model evaluation...');

        const { features, labels } = this.testData;
        const xs = tf.tensor2d(features);
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

        const evaluation = this.model.evaluate(xs, ys);
        const loss = evaluation[0].dataSync()[0];
        const accuracy = evaluation[1].dataSync()[0];

        this.log(`Evaluation results - Accuracy: ${(accuracy * 100).toFixed(2)}%, Loss: ${loss.toFixed(4)}`);

        xs.dispose();
        ys.dispose();
        evaluation.forEach(tensor => tensor.dispose());
    }

    async testRandomSamples() {
        if (!this.model || !this.testData) {
            this.log('Error: No model or test data available');
            return;
        }

        const { features, labels, rawData } = this.testData;
        const resultsContainer = document.getElementById('randomTestResults');
        resultsContainer.innerHTML = '';

        // Randomly select 5 samples
        const indices = [];
        for (let i = 0; i < Math.min(5, features.length); i++) {
            indices.push(Math.floor(Math.random() * features.length));
        }

        for (const index of indices) {
            const feature = features[index];
            const trueLabel = labels[index];
            const sequence = rawData[index].Sequence || 'N/A';

            const inputTensor = tf.tensor2d([feature]);
            const prediction = this.model.predict(inputTensor);
            const results = await prediction.data();
            
            const predictedClass = this.classLabels[results.indexOf(Math.max(...results))];
            const trueClass = this.classLabels[trueLabel];
            const confidence = Math.max(...results) * 100;

            const resultDiv = document.createElement('div');
            resultDiv.className = `random-test-item ${predictedClass === trueClass ? '' : 'error'}`;
            resultDiv.innerHTML = `
                <strong>Sample ${index + 1}</strong><br>
                <small>Sequence: ${sequence.substring(0, 50)}${sequence.length > 50 ? '...' : ''}</small><br>
                True Label: ${trueClass} | Predicted: ${predictedClass}<br>
                Confidence: ${confidence.toFixed(2)}%<br>
                ${predictedClass === trueClass ? '✅ Correct' : '❌ Incorrect'}
            `;

            resultsContainer.appendChild(resultDiv);
            inputTensor.dispose();
            prediction.dispose();
        }

        this.log('Random testing completed');
    }

    async testSingleSequence() {
        const sequenceInput = document.getElementById('singleSequence').value.trim().toUpperCase();
        
        if (!sequenceInput) {
            this.log('Error: Please enter a DNA sequence');
            return;
        }

        if (!/^[ATCG]+$/.test(sequenceInput)) {
            this.log('Error: DNA sequence can only contain A, T, C, G characters');
            return;
        }

        if (!this.model) {
            this.log('Error: No model available, please train or load a model first');
            return;
        }

        try {
            // Extract features
            const features = DataLoader.extractFeaturesFromSequence(sequenceInput);
            const inputTensor = tf.tensor2d([features]);
            const prediction = this.model.predict(inputTensor);
            const results = await prediction.data();
            
            const maxConfidence = Math.max(...results);
            const predictedClass = this.classLabels[results.indexOf(maxConfidence)];
            
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
            `;

            this.log(`Single sequence test completed: ${predictedClass} (${(maxConfidence * 100).toFixed(2)}%)`);
            
            inputTensor.dispose();
            prediction.dispose();
        } catch (error) {
            this.log(`Single sequence test error: ${error.message}`);
        }
    }

    async saveModel() {
        if (!this.model) {
            this.log('Error: No model to save');
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
            modelJsonLink.download = 'dna-model.json';
            modelJsonLink.click();

            // Save weights
            await this.model.save('downloads://dna-model');

            this.log('Model saved successfully! Check your downloads folder for dna-model.json and dna-model.weights.bin');
        } catch (error) {
            this.log(`Model save error: ${error.message}`);
        }
    }

    async loadModel() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];

        if (!jsonFile || !weightsFile) {
            this.log('Error: Please select both model JSON and weights files');
            return;
        }

        this.log('Loading model...');

        try {
            const modelJson = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(JSON.parse(reader.result));
                reader.onerror = reject;
                reader.readAsText(jsonFile);
            });

            const modelWeights = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsArrayBuffer(weightsFile);
            });

            this.model = await tf.loadLayersModel(tf.io.fromMemory(modelJson, modelWeights));
            this.log('Model loaded successfully!');
            this.updateModelInfo();
        } catch (error) {
            this.log(`Model load error: ${error.message}`);
        }
    }

    updateModelInfo() {
        if (!this.model) return;

        let totalParams = 0;
        this.model.summary(null, null, (line) => {
            const match = line.match(/params: (\d+)/);
            if (match) {
                totalParams += parseInt(match[1]);
            }
        });

        document.getElementById('layersCount').textContent = this.model.layers.length;
        document.getElementById('totalParams').textContent = totalParams.toLocaleString();
    }

    resetSystem() {
        this.model = null;
        this.trainData = null;
        this.testData = null;
        
        document.getElementById('trainSamples').textContent = '0';
        document.getElementById('testSamples').textContent = '0';
        document.getElementById('layersCount').textContent = '0';
        document.getElementById('totalParams').textContent = '0';
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

        this.log('System reset');
    }
}

// Initialize application
let dnaClassifier;
document.addEventListener('DOMContentLoaded', () => {
    dnaClassifier = new DNAClassifier();
});
