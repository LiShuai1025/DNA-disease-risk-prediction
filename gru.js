class DiseaseRiskModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.trainingHistory = null;
        this.modelInfo = {
            inputSize: 100,
            numericalFeatures: 7,
            hiddenUnits: 64,
            version: '2.0'
        };
    }

    // Create LSTM model
    async createModel() {
        try {
            // Sequence input branch
            const sequenceInput = tf.input({shape: [this.modelInfo.inputSize], name: 'sequence_input'});
            
            const embedding = tf.layers.embedding({
                inputDim: 4,
                outputDim: 32,
                inputLength: this.modelInfo.inputSize
            }).apply(sequenceInput);
            
            const lstmLayer = tf.layers.lstm({
                units: this.modelInfo.hiddenUnits,
                dropout: 0.2,
                recurrentDropout: 0.2,
                returnSequences: false
            }).apply(embedding);
            
            // Numerical features input branch
            const numericalInput = tf.input({shape: [this.modelInfo.numericalFeatures], name: 'numerical_input'});
            const numericalDense = tf.layers.dense({
                units: 16,
                activation: 'relu'
            }).apply(numericalInput);
            
            // Merge branches
            const concatenated = tf.layers.concatenate().apply([lstmLayer, numericalDense]);
            
            const dense1 = tf.layers.dense({
                units: 32,
                activation: 'relu'
            }).apply(concatenated);
            
            const dropout = tf.layers.dropout({rate: 0.3}).apply(dense1);
            
            // Output layer
            const output = tf.layers.dense({
                units: 3,
                activation: 'softmax',
                name: 'output'
            }).apply(dropout);
            
            this.model = tf.model({
                inputs: [sequenceInput, numericalInput],
                outputs: output
            });
            
            this.model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            console.log('LSTM model created successfully');
            return this.model;
            
        } catch (error) {
            console.error('Error creating model:', error);
            throw error;
        }
    }

    // Train the model
    async trainModel(sequences, numericalFeatures, labels, epochs = 50) {
        if (!this.model) {
            await this.createModel();
        }

        try {
            // Prepare data
            const sequenceTensor = tf.tensor2d(sequences, [sequences.length, this.modelInfo.inputSize]);
            const featuresTensor = tf.tensor2d(numericalFeatures, [numericalFeatures.length, this.modelInfo.numericalFeatures]);
            const labelsTensor = tf.oneHot(labels, 3);

            // Train model
            const history = await this.model.fit([sequenceTensor, featuresTensor], labelsTensor, {
                epochs: epochs,
                batchSize: 16,
                validationSplit: 0.2,
                verbose: 0,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        const progress = ((epoch + 1) / epochs * 100).toFixed(1);
                        updateStatus(`Training: ${progress}% complete - Loss: ${logs.loss.toFixed(4)}`);
                    }
                }
            });

            // Clean up tensors
            sequenceTensor.dispose();
            featuresTensor.dispose();
            labelsTensor.dispose();

            this.isTrained = true;
            this.trainingHistory = history;
            
            console.log('Model training completed');
            return history;
            
        } catch (error) {
            console.error('Error training model:', error);
            throw error;
        }
    }

    // Batch prediction
    async predictBatch(sequences, numericalFeatures) {
        if (!this.isTrained) {
            throw new Error('Model not trained');
        }

        try {
            const sequenceTensor = tf.tensor2d(sequences, [sequences.length, this.modelInfo.inputSize]);
            const featuresTensor = tf.tensor2d(numericalFeatures, [numericalFeatures.length, this.modelInfo.numericalFeatures]);
            
            const predictions = this.model.predict([sequenceTensor, featuresTensor]);
            const predictionData = await predictions.data();
            
            // Clean up tensors
            sequenceTensor.dispose();
            featuresTensor.dispose();
            predictions.dispose();
            
            // Convert prediction data to risk levels
            const results = [];
            for (let i = 0; i < sequences.length; i++) {
                const startIdx = i * 3;
                const probs = Array.from(predictionData.slice(startIdx, startIdx + 3));
                const predictedClass = probs.indexOf(Math.max(...probs));
                
                results.push({
                    predictedRisk: dataLoader.riskLabels[predictedClass],
                    confidence: Math.max(...probs),
                    probabilities: probs
                });
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in batch prediction:', error);
            throw error;
        }
    }

    // Get model information
    getModelInfo() {
        return this.modelInfo;
    }

    // Get training history
    getTrainingHistory() {
        return this.trainingHistory;
    }
}

const diseaseModel = new DiseaseRiskModel();
