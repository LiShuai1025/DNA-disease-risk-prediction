class DiseaseRiskModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.trainingHistory = null;
        this.modelInfo = {
            inputSize: 100,
            numericalFeatures: 7,
            hiddenUnits: 32, // Reduced for performance
            version: '2.1'
        };
    }

    // Create LSTM model (simplified for performance)
    async createModel() {
        try {
            // Simplified model architecture
            const sequenceInput = tf.input({shape: [this.modelInfo.inputSize], name: 'sequence_input'});
            
            const embedding = tf.layers.embedding({
                inputDim: 4,
                outputDim: 16, // Reduced embedding
                inputLength: this.modelInfo.inputSize
            }).apply(sequenceInput);
            
            const lstmLayer = tf.layers.lstm({
                units: this.modelInfo.hiddenUnits,
                dropout: 0.1, // Reduced dropout
                recurrentDropout: 0.1,
                returnSequences: false
            }).apply(embedding);
            
            const numericalInput = tf.input({shape: [this.modelInfo.numericalFeatures], name: 'numerical_input'});
            const numericalDense = tf.layers.dense({
                units: 8, // Reduced units
                activation: 'relu'
            }).apply(numericalInput);
            
            const concatenated = tf.layers.concatenate().apply([lstmLayer, numericalDense]);
            
            const dense1 = tf.layers.dense({
                units: 16, // Reduced units
                activation: 'relu'
            }).apply(concatenated);
            
            const dropout = tf.layers.dropout({rate: 0.2}).apply(dense1);
            
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
            
            console.log('Optimized LSTM model created successfully');
            return this.model;
            
        } catch (error) {
            console.error('Error creating model:', error);
            throw error;
        }
    }

    // Train model with progress callback
    async trainModel(sequences, numericalFeatures, labels, epochs = 20, progressCallback = null) {
        if (!this.model) {
            await this.createModel();
        }

        try {
            const sequenceTensor = tf.tensor2d(sequences, [sequences.length, this.modelInfo.inputSize]);
            const featuresTensor = tf.tensor2d(numericalFeatures, [numericalFeatures.length, this.modelInfo.numericalFeatures]);
            const labelsTensor = tf.oneHot(labels, 3);

            // Custom training loop for progress updates
            const batchSize = 8; // Smaller batch size
            const numBatches = Math.ceil(sequences.length / batchSize);
            
            for (let epoch = 0; epoch < epochs; epoch++) {
                let totalLoss = 0;
                
                for (let i = 0; i < numBatches; i++) {
                    const start = i * batchSize;
                    const end = Math.min(start + batchSize, sequences.length);
                    
                    const batchSequences = sequenceTensor.slice([start, 0], [end - start, -1]);
                    const batchFeatures = featuresTensor.slice([start, 0], [end - start, -1]);
                    const batchLabels = labelsTensor.slice([start, 0], [end - start, -1]);
                    
                    const history = await this.model.trainOnBatch([batchSequences, batchFeatures], batchLabels);
                    totalLoss += history[0];
                    
                    // Clean up batch tensors
                    tf.dispose([batchSequences, batchFeatures, batchLabels]);
                    
                    // Update progress
                    if (progressCallback) {
                        const batchProgress = ((epoch * numBatches + i + 1) / (epochs * numBatches)) * 100;
                        progressCallback(Math.min(100, Math.round(batchProgress)));
                    }
                    
                    // Yield to prevent blocking
                    await tf.nextFrame();
                }
                
                console.log(`Epoch ${epoch + 1}/${epochs}, Loss: ${(totalLoss / numBatches).toFixed(4)}`);
            }

            // Clean up tensors
            tf.dispose([sequenceTensor, featuresTensor, labelsTensor]);

            this.isTrained = true;
            console.log('Model training completed');
            
        } catch (error) {
            console.error('Error training model:', error);
            throw error;
        }
    }

    // Batch prediction with progress
    async predictBatch(sequences, numericalFeatures, progressCallback = null) {
        if (!this.isTrained) {
            throw new Error('Model not trained');
        }

        try {
            const results = [];
            const batchSize = 10; // Smaller batches for responsiveness
            
            for (let i = 0; i < sequences.length; i += batchSize) {
                const batchSequences = sequences.slice(i, i + batchSize);
                const batchFeatures = numericalFeatures.slice(i, i + batchSize);
                
                const sequenceTensor = tf.tensor2d(batchSequences, [batchSequences.length, this.modelInfo.inputSize]);
                const featuresTensor = tf.tensor2d(batchFeatures, [batchFeatures.length, this.modelInfo.numericalFeatures]);
                
                const predictions = this.model.predict([sequenceTensor, featuresTensor]);
                const predictionData = await predictions.data();
                
                // Process batch results
                for (let j = 0; j < batchSequences.length; j++) {
                    const startIdx = j * 3;
                    const probs = Array.from(predictionData.slice(startIdx, startIdx + 3));
                    const predictedClass = probs.indexOf(Math.max(...probs));
                    
                    results.push({
                        predictedRisk: dataLoader.riskLabels[predictedClass],
                        confidence: Math.max(...probs),
                        probabilities: probs
                    });
                }
                
                // Clean up
                tf.dispose([sequenceTensor, featuresTensor, predictions]);
                
                // Update progress
                if (progressCallback) {
                    const progress = ((i + batchSize) / sequences.length) * 100;
                    progressCallback(Math.min(100, Math.round(progress)));
                }
                
                // Yield to prevent blocking
                await tf.nextFrame();
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in batch prediction:', error);
            throw error;
        }
    }

    getModelInfo() {
        return this.modelInfo;
    }

    getTrainingHistory() {
        return this.trainingHistory;
    }
}

const diseaseModel = new DiseaseRiskModel();
