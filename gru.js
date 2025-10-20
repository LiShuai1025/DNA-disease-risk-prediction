class GRUDiseaseModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.maxSequenceLength = 100;
        this.vocabSize = 4; // A, T, C, G
        this.numOutputs = 2; // Binary classification outputs
        this.trainingInProgress = false;
    }

    // Convert DNA sequence to numerical encoding
    encodeSequence(sequence) {
        const encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        const encoded = [];
        
        const effectiveLength = Math.min(sequence.length, this.maxSequenceLength);
        
        for (let i = 0; i < effectiveLength; i++) {
            const char = sequence[i].toUpperCase();
            encoded.push(encoding[char] || 0);
        }
        
        // Pad sequence if shorter than max length
        while (encoded.length < this.maxSequenceLength) {
            encoded.push(0);
        }
        
        return encoded;
    }

    // Convert to one-hot encoding for GRU input
    oneHotEncodeBatch(encodedSequences) {
        const batchSize = encodedSequences.length;
        const oneHotBatch = [];
        
        for (let i = 0; i < batchSize; i++) {
            const oneHot = [];
            for (let j = 0; j < this.maxSequenceLength; j++) {
                const vector = new Array(this.vocabSize).fill(0);
                vector[encodedSequences[i][j]] = 1;
                oneHot.push(vector);
            }
            oneHotBatch.push(oneHot);
        }
        
        return oneHotBatch;
    }

    // Prepare training data with batching for memory efficiency
    prepareTrainingData(samples) {
        const sequences = [];
        const labels = [];
        
        // Limit dataset size for browser performance
        const maxSamples = Math.min(samples.length, 200);
        
        for (let i = 0; i < maxSamples; i++) {
            const sample = samples[i];
            if (sample.sequence && sample.actualRisk) {
                const encoded = this.encodeSequence(sample.sequence);
                sequences.push(encoded);
                
                // Binary classification: High risk vs Low/Medium
                const isHighRisk = sample.actualRisk === 'High' ? 1 : 0;
                const isPathogenic = this.isPathogenicSequence(sample) ? 1 : 0;
                
                labels.push([isHighRisk, isPathogenic]);
            }
        }
        
        console.log(`Prepared ${sequences.length} samples for training`);
        
        return {
            sequences: sequences,
            labels: labels
        };
    }

    // Enhanced feature detection for pathogenic sequences
    isPathogenicSequence(sample) {
        const features = sample.features;
        if (!features) return false;
        
        const hasHighGC = features.gcContent > 60;
        const hasLowComplexity = features.kmerFreq < 0.3;
        const hasBaseBias = this.calculateBaseBias(sample) > 0.35;
        
        return hasHighGC || hasLowComplexity || hasBaseBias;
    }

    calculateBaseBias(sample) {
        const bases = [sample.features.numA, sample.features.numT, sample.features.numC, sample.features.numG];
        const maxBase = Math.max(...bases);
        const total = bases.reduce((a, b) => a + b, 0);
        return maxBase / total;
    }

    // Build optimized GRU model
    buildModel() {
        const model = tf.sequential();
        
        // GRU Layer
        model.add(tf.layers.gru({
            units: 32,
            returnSequences: false,
            inputShape: [this.maxSequenceLength, this.vocabSize],
            dropout: 0.1,
            recurrentDropout: 0.1
        }));
        
        // Dense layers
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.001})
        }));
        
        model.add(tf.layers.dropout({rate: 0.2}));
        
        model.add(tf.layers.dense({
            units: 8,
            activation: 'relu'
        }));
        
        // Multi-output layer
        model.add(tf.layers.dense({
            units: this.numOutputs,
            activation: 'sigmoid'
        }));
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        this.model = model;
        console.log('GRU Model built successfully');
        return model;
    }

    // Train the GRU model with performance optimizations
    async trainModel(samples, progressCallback = null) {
        if (this.trainingInProgress) {
            throw new Error('Training already in progress');
        }
        
        this.trainingInProgress = true;
        
        try {
            if (progressCallback) progressCallback(10, 'Building GRU model...');
            
            // Clear any existing model to free memory
            if (this.model) {
                this.model.dispose();
            }
            
            // Build model
            this.buildModel();
            
            if (progressCallback) progressCallback(20, 'Preparing training data...');
            
            // Prepare training data
            const {sequences, labels} = this.prepareTrainingData(samples);
            
            // Convert to tensors
            const sequencesTensor = tf.tensor3d(this.oneHotEncodeBatch(sequences));
            const labelsTensor = tf.tensor2d(labels);
            
            if (progressCallback) progressCallback(40, 'Starting training...');
            
            // Train model
            await this.model.fit(sequencesTensor, labelsTensor, {
                epochs: 20, // Further reduced for stability
                batchSize: 16,
                validationSplit: 0.2,
                verbose: 0,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        await new Promise(resolve => setTimeout(resolve, 10));
                        
                        const progress = 40 + (epoch / 20) * 50;
                        if (progressCallback) {
                            progressCallback(
                                Math.min(90, progress), 
                                `Epoch ${epoch + 1}/20 - Loss: ${logs.loss.toFixed(4)}`
                            );
                        }
                    }
                }
            });
            
            if (progressCallback) progressCallback(100, 'Training completed');
            
            this.isTrained = true;
            console.log('GRU Model training completed');
            
            // Clean up tensors
            sequencesTensor.dispose();
            labelsTensor.dispose();
            
        } catch (error) {
            console.error('Error training GRU model:', error);
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            throw error;
        } finally {
            this.trainingInProgress = false;
        }
    }

    // NEW: Simplified prediction method that avoids the variable scope issue
    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained || !this.model) {
            throw new Error('Model not trained or available');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            // Process samples one by one for stability
            for (let i = 0; i < totalSamples; i++) {
                const sample = samples[i];
                const result = await this.predictSingleSample(sample);
                results.push(result);
                
                if (progressCallback) {
                    const progress = ((i + 1) / totalSamples) * 100;
                    progressCallback(Math.round(progress));
                }
                
                // Small delay to prevent blocking
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in prediction:', error);
            throw error;
        }
    }

    // NEW: Completely rewritten single sample prediction
    async predictSingleSample(sample) {
        if (!this.model) {
            return this.getDefaultPrediction();
        }

        try {
            // Encode sequence
            const encoded = this.encodeSequence(sample.sequence || '');
            const oneHot = this.oneHotEncodeBatch([encoded]);
            const tensor = tf.tensor3d(oneHot);
            
            // Make prediction
            const prediction = this.model.predict(tensor);
            const values = await prediction.data();
            
            // Extract probabilities
            const highRiskProb = values[0];
            const pathogenicProb = values[1];
            const confidence = Math.max(highRiskProb, pathogenicProb);
            
            // Determine risk level
            let predictedRisk = 'Low';
            if (highRiskProb > 0.7) {
                predictedRisk = 'High';
            } else if (highRiskProb > 0.4 || pathogenicProb > 0.6) {
                predictedRisk = 'Medium';
            }
            
            // Clean up tensors
            tensor.dispose();
            prediction.dispose();
            
            return {
                predictedRisk: predictedRisk,
                confidence: confidence,
                highRiskProbability: highRiskProb,
                pathogenicProbability: pathogenicProb
            };
            
        } catch (error) {
            console.error('Error predicting single sample:', error);
            return this.getDefaultPrediction();
        }
    }

    // Helper method for default prediction when model fails
    getDefaultPrediction() {
        // Simple rule-based fallback
        return {
            predictedRisk: 'Medium',
            confidence: 0.5,
            highRiskProbability: 0.5,
            pathogenicProbability: 0.5
        };
    }

    // Get model information
    getModelInfo() {
        return {
            version: '3.0-GRU-Stable',
            type: 'Multi-Output GRU Neural Network',
            architecture: 'GRU(32) -> Dense(16) -> Dense(8) -> Output(2)',
            outputs: ['High_Risk_Probability', 'Pathogenic_Probability']
        };
    }
}

const diseaseModel = new GRUDiseaseModel();
