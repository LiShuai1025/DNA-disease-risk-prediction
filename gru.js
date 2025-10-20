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
        
        // Use only first maxSequenceLength characters for performance
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

    // Convert to one-hot encoding for GRU input - optimized version
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
            sequences: tf.tensor3d(this.oneHotEncodeBatch(sequences)),
            labels: tf.tensor2d(labels)
        };
    }

    // Enhanced feature detection for pathogenic sequences
    isPathogenicSequence(sample) {
        const features = sample.features;
        if (!features) return false;
        
        // Simple rules based on known pathogenic markers
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
        
        // Smaller GRU Layer for browser performance
        model.add(tf.layers.gru({
            units: 32, // Reduced from 64
            returnSequences: false,
            inputShape: [this.maxSequenceLength, this.vocabSize],
            dropout: 0.1, // Reduced regularization
            recurrentDropout: 0.1
        }));
        
        // Smaller dense layers
        model.add(tf.layers.dense({
            units: 16, // Reduced from 32
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.001}) // Reduced regularization
        }));
        
        model.add(tf.layers.dropout({rate: 0.2})); // Reduced dropout
        
        model.add(tf.layers.dense({
            units: 8, // Reduced from 16
            activation: 'relu'
        }));
        
        // Multi-output layer
        model.add(tf.layers.dense({
            units: this.numOutputs,
            activation: 'sigmoid'
        }));
        
        // Compile model with simpler optimizer
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        this.model = model;
        console.log('Optimized GRU Model built successfully');
        return model;
    }

    // Train the GRU model with performance optimizations
    async trainModel(samples, progressCallback = null) {
        if (this.trainingInProgress) {
            throw new Error('Training already in progress');
        }
        
        this.trainingInProgress = true;
        
        try {
            if (progressCallback) progressCallback(10, 'Building optimized GRU model...');
            
            // Clear any existing model to free memory
            if (this.model) {
                this.model.dispose();
            }
            
            // Build optimized model
            this.buildModel();
            
            if (progressCallback) progressCallback(20, 'Preparing training data...');
            
            // Prepare training data with size limits
            const {sequences, labels} = this.prepareTrainingData(samples);
            
            if (progressCallback) progressCallback(40, 'Starting training (this may take a while)...');
            
            // Train with fewer epochs and smaller batch size
            const history = await this.model.fit(sequences, labels, {
                epochs: 30, // Reduced from 50
                batchSize: 16, // Smaller batch size
                validationSplit: 0.2,
                verbose: 0, // Reduce console output
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        // Give browser time to process UI events
                        await new Promise(resolve => setTimeout(resolve, 10));
                        
                        const progress = 40 + (epoch / 30) * 50;
                        if (progressCallback) {
                            progressCallback(
                                Math.min(90, progress), 
                                `Epoch ${epoch + 1}/30 - Loss: ${logs.loss.toFixed(4)}`
                            );
                        }
                    }
                }
            });
            
            if (progressCallback) progressCallback(100, 'Training completed');
            
            this.isTrained = true;
            console.log('GRU Model training completed');
            
            // Clean up tensors immediately
            sequences.dispose();
            labels.dispose();
            
        } catch (error) {
            console.error('Error training GRU model:', error);
            // Ensure model is disposed on error
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            throw error;
        } finally {
            this.trainingInProgress = false;
        }
    }

    // Predict samples using GRU model with batching
    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained) {
            throw new Error('Model not trained');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            // Process in smaller batches to avoid memory issues
            const batchSize = 10;
            
            for (let i = 0; i < totalSamples; i += batchSize) {
                const batchSamples = samples.slice(i, i + batchSize);
                const batchResults = await this.predictBatch(batchSamples);
                results.push(...batchResults);
                
                if (progressCallback) {
                    const progress = ((i + batchSize) / totalSamples) * 100;
                    progressCallback(Math.min(100, Math.round(progress)));
                }
                
                // Give browser time to process UI events
                await new Promise(resolve => setTimeout(resolve, 50));
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in GRU prediction:', error);
            throw error;
        }
    }

    // Predict batch of samples
    async predictBatch(samples) {
        const batchResults = [];
        const sequences = [];
        
        // Encode all sequences in batch
        for (const sample of samples) {
            if (sample.sequence) {
                const encoded = this.encodeSequence(sample.sequence);
                sequences.push(encoded);
            } else {
                sequences.push(new Array(this.maxSequenceLength).fill(0));
            }
        }
        
        // Convert to one-hot in batch
        const oneHotBatch = this.oneHotEncodeBatch(sequences);
        const tensor = tf.tensor3d(oneHotBatch);
        
        try {
            const prediction = this.model.predict(tensor);
            const values = await prediction.data();
            
            // Process results
            for (let i = 0; i < samples.length; i++) {
                const highRiskProb = values[i * 2];
                const pathogenicProb = values[i * 2 + 1];
                const confidence = Math.max(highRiskProb, pathogenicProb);
                
                // Determine final risk classification
                let predictedRisk = 'Low';
                if (highRiskProb > 0.7) {
                    predictedRisk = 'High';
                } else if (highRiskProb > 0.4 || pathogenicProb > 0.6) {
                    predictedRisk = 'Medium';
                }
                
                batchResults.push({
                    predictedRisk: predictedRisk,
                    confidence: confidence,
                    highRiskProbability: highRiskProb,
                    pathogenicProbability: pathogenicProb
                });
            }
            
            return batchResults;
            
        } finally {
            // Clean up tensors
            tensor.dispose();
            if (prediction) {
                prediction.dispose();
            }
        }
    }

    // Get model information
    getModelInfo() {
        return {
            version: '2.0-GRU-Optimized',
            type: 'Multi-Output GRU Neural Network',
            architecture: 'GRU(32) -> Dense(16) -> Dense(8) -> Output(2)',
            outputs: ['High_Risk_Probability', 'Pathogenic_Probability']
        };
    }
}

const diseaseModel = new GRUDiseaseModel();
