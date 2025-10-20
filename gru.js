class GRUDiseaseModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.maxSequenceLength = 100;
        this.vocabSize = 4; // A, T, C, G
        this.numOutputs = 2; // Binary classification outputs
    }

    // Convert DNA sequence to numerical encoding
    encodeSequence(sequence) {
        const encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        const encoded = [];
        
        for (let i = 0; i < Math.min(sequence.length, this.maxSequenceLength); i++) {
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
    oneHotEncode(encodedSequence) {
        const oneHot = [];
        for (let i = 0; i < encodedSequence.length; i++) {
            const vector = new Array(this.vocabSize).fill(0);
            vector[encodedSequence[i]] = 1;
            oneHot.push(vector);
        }
        return oneHot;
    }

    // Prepare training data
    prepareTrainingData(samples) {
        const sequences = [];
        const labels = [];
        
        samples.forEach(sample => {
            if (sample.sequence && sample.actualRisk) {
                const encoded = this.encodeSequence(sample.sequence);
                const oneHot = this.oneHotEncode(encoded);
                sequences.push(oneHot);
                
                // Binary classification: High risk vs Low/Medium
                const isHighRisk = sample.actualRisk === 'High' ? 1 : 0;
                const isPathogenic = this.isPathogenicSequence(sample) ? 1 : 0;
                
                labels.push([isHighRisk, isPathogenic]);
            }
        });
        
        return {
            sequences: tf.tensor3d(sequences),
            labels: tf.tensor2d(labels)
        };
    }

    // Enhanced feature detection for pathogenic sequences
    isPathogenicSequence(sample) {
        const features = sample.features;
        if (!features) return false;
        
        // Rules based on known pathogenic markers
        const hasHighGC = features.gcContent > 60;
        const hasLowComplexity = features.kmerFreq < 0.3;
        const hasBaseBias = this.calculateBaseBias(sample) > 0.35;
        const hasRepeats = this.detectRepeats(sample.sequence) > 0.1;
        
        return hasHighGC || hasLowComplexity || hasBaseBias || hasRepeats;
    }

    calculateBaseBias(sample) {
        const bases = [sample.features.numA, sample.features.numT, sample.features.numC, sample.features.numG];
        const maxBase = Math.max(...bases);
        const total = bases.reduce((a, b) => a + b, 0);
        return maxBase / total;
    }

    detectRepeats(sequence) {
        if (!sequence || sequence.length < 6) return 0;
        
        let repeatCount = 0;
        for (let i = 0; i < sequence.length - 5; i++) {
            const kmer = sequence.substring(i, i + 3);
            let repeats = 0;
            for (let j = i + 3; j < sequence.length - 2; j += 3) {
                if (sequence.substring(j, j + 3) === kmer) {
                    repeats++;
                } else {
                    break;
                }
            }
            if (repeats >= 2) repeatCount++;
        }
        
        return repeatCount / (sequence.length / 3);
    }

    // Build the GRU model
    buildModel() {
        const model = tf.sequential();
        
        // GRU Layer
        model.add(tf.layers.gru({
            units: 64,
            returnSequences: false,
            inputShape: [this.maxSequenceLength, this.vocabSize],
            dropout: 0.2,
            recurrentDropout: 0.2
        }));
        
        // Dense layers
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.01})
        }));
        
        model.add(tf.layers.dropout({rate: 0.3}));
        
        model.add(tf.layers.dense({
            units: 16,
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
            metrics: ['accuracy', 'precision', 'recall']
        });
        
        this.model = model;
        console.log('GRU Model built successfully');
        return model;
    }

    // Train the GRU model
    async trainModel(samples, progressCallback = null) {
        try {
            if (progressCallback) progressCallback(10, 'Building GRU model...');
            
            // Build model if not already built
            if (!this.model) {
                this.buildModel();
            }
            
            if (progressCallback) progressCallback(20, 'Preparing training data...');
            
            // Prepare training data
            const {sequences, labels} = this.prepareTrainingData(samples);
            
            if (progressCallback) progressCallback(40, 'Training GRU model...');
            
            // Train the model
            const history = await this.model.fit(sequences, labels, {
                epochs: 50,
                batchSize: 32,
                validationSplit: 0.2,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        const progress = 40 + (epoch / 50) * 50;
                        if (progressCallback) {
                            progressCallback(
                                Math.min(90, progress), 
                                `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}`
                            );
                        }
                    }
                }
            });
            
            if (progressCallback) progressCallback(100, 'Training completed');
            
            this.isTrained = true;
            console.log('GRU Model training completed');
            
            // Clean up tensors
            sequences.dispose();
            labels.dispose();
            
        } catch (error) {
            console.error('Error training GRU model:', error);
            throw error;
        }
    }

    // Predict samples using GRU model
    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained) {
            throw new Error('Model not trained');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            for (let i = 0; i < totalSamples; i++) {
                const sample = samples[i];
                const result = await this.predictSingleSample(sample);
                results.push(result);
                
                if (progressCallback) {
                    const progress = ((i + 1) / totalSamples) * 100;
                    progressCallback(Math.round(progress));
                }
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in GRU prediction:', error);
            throw error;
        }
    }

    // Predict single sample
    async predictSingleSample(sample) {
        if (!sample.sequence) {
            return {
                predictedRisk: 'Low',
                confidence: 0.5,
                probabilities: [0.5, 0.5]
            };
        }
        
        const encoded = this.encodeSequence(sample.sequence);
        const oneHot = this.oneHotEncode(encoded);
        const tensor = tf.tensor3d([oneHot]);
        
        const prediction = this.model.predict(tensor);
        const values = await prediction.data();
        
        // Clean up tensors
        tensor.dispose();
        prediction.dispose();
        
        const [highRiskProb, pathogenicProb] = values;
        const confidence = Math.max(highRiskProb, pathogenicProb);
        
        // Determine final risk classification
        let predictedRisk = 'Low';
        if (highRiskProb > 0.7) {
            predictedRisk = 'High';
        } else if (highRiskProb > 0.4 || pathogenicProb > 0.6) {
            predictedRisk = 'Medium';
        }
        
        return {
            predictedRisk: predictedRisk,
            confidence: confidence,
            probabilities: [highRiskProb, pathogenicProb],
            highRiskProbability: highRiskProb,
            pathogenicProbability: pathogenicProb
        };
    }

    // Get model information
    getModelInfo() {
        return {
            version: '2.0-GRU',
            type: 'Multi-Output GRU Neural Network',
            architecture: 'GRU(64) -> Dense(32) -> Dense(16) -> Output(2)',
            outputs: ['High_Risk_Probability', 'Pathogenic_Probability']
        };
    }

    // Model summary
    async summary() {
        if (this.model) {
            this.model.summary();
        }
    }
}

const diseaseModel = new GRUDiseaseModel();
