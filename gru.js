class SimpleGRUModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.maxSequenceLength = 100;
        this.vocabSize = 4;
        this.numOutputs = 3;
        this.trainingInProgress = false;
    }

    encodeSequence(sequence) {
        if (!sequence || typeof sequence !== 'string') {
            return new Array(this.maxSequenceLength).fill(0);
        }

        const encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        const encoded = [];
        const seq = sequence.toUpperCase();
        
        for (let i = 0; i < Math.min(seq.length, this.maxSequenceLength); i++) {
            encoded.push(encoding[seq[i]] || 0);
        }
        
        // 填充
        while (encoded.length < this.maxSequenceLength) {
            encoded.push(0);
        }
        
        return encoded;
    }

    oneHotEncodeBatch(encodedSequences) {
        const oneHotBatch = [];
        
        for (let i = 0; i < encodedSequences.length; i++) {
            const oneHot = [];
            for (let j = 0; j < this.maxSequenceLength; j++) {
                const vector = new Array(this.vocabSize).fill(0);
                const baseIndex = encodedSequences[i][j];
                if (baseIndex < this.vocabSize) {
                    vector[baseIndex] = 1;
                }
                oneHot.push(vector);
            }
            oneHotBatch.push(oneHot);
        }
        
        return oneHotBatch;
    }

    buildSimpleModel() {
        try {
            const model = tf.sequential();
            
            // 简化模型结构
            model.add(tf.layers.gru({
                units: 32,
                inputShape: [this.maxSequenceLength, this.vocabSize],
                dropout: 0.2
            }));
            
            model.add(tf.layers.dense({
                units: 16,
                activation: 'relu'
            }));
            
            model.add(tf.layers.dropout({rate: 0.3}));
            
            // 输出层
            model.add(tf.layers.dense({
                units: this.numOutputs,
                activation: 'softmax'
            }));
            
            model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            this.model = model;
            console.log('Simple GRU Model built successfully');
            return model;
        } catch (error) {
            console.error('Error building model:', error);
            throw error;
        }
    }

    prepareTrainingData(samples) {
        const sequences = [];
        const labels = [];
        
        // 使用更多样本
        const maxSamples = Math.min(samples.length, 200);
        
        for (let i = 0; i < maxSamples; i++) {
            const sample = samples[i];
            if (sample.sequence && sample.actualRisk) {
                sequences.push(sample.sequence);
                
                const labelIndex = {'High': 0, 'Medium': 1, 'Low': 2}[sample.actualRisk];
                if (labelIndex !== undefined) {
                    const oneHot = new Array(3).fill(0);
                    oneHot[labelIndex] = 1;
                    labels.push(oneHot);
                }
            }
        }
        
        const encodedSequences = sequences.map(seq => this.encodeSequence(seq));
        const oneHotSequences = this.oneHotEncodeBatch(encodedSequences);
        
        return {
            sequences: tf.tensor3d(oneHotSequences),
            labels: tf.tensor2d(labels)
        };
    }

    async trainModel(samples, progressCallback = null) {
        if (this.trainingInProgress) {
            throw new Error('Training already in progress');
        }
        
        this.trainingInProgress = true;
        
        let sequencesTensor, labelsTensor;
        
        try {
            if (progressCallback) progressCallback(10, 'Building simple model...');
            
            if (this.model) {
                this.model.dispose();
            }
            
            this.buildSimpleModel();
            
            if (progressCallback) progressCallback(30, 'Preparing training data...');
            
            const trainingData = this.prepareTrainingData(samples);
            sequencesTensor = trainingData.sequences;
            labelsTensor = trainingData.labels;
            
            if (progressCallback) progressCallback(50, 'Starting training...');
            
            // 简化训练过程
            await this.model.fit(sequencesTensor, labelsTensor, {
                epochs: 30,
                batchSize: 8,
                validationSplit: 0.2,
                verbose: 0,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        if (progressCallback) {
                            const progress = 50 + (epoch / 30) * 45;
                            const message = `Epoch ${epoch + 1}/30 - Loss: ${logs.loss.toFixed(4)}`;
                            progressCallback(progress, message);
                        }
                    }
                }
            });
            
            if (progressCallback) progressCallback(100, 'Training completed');
            
            this.isTrained = true;
            console.log('Model training completed');
            
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        } finally {
            if (sequencesTensor) sequencesTensor.dispose();
            if (labelsTensor) labelsTensor.dispose();
            this.trainingInProgress = false;
        }
    }

    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained || !this.model) {
            throw new Error('Model not trained');
        }

        const results = [];
        const batchSize = 8;
        
        for (let i = 0; i < samples.length; i += batchSize) {
            const batchSamples = samples.slice(i, i + batchSize);
            const batchResults = await this.predictBatch(batchSamples);
            results.push(...batchResults);
            
            if (progressCallback) {
                const progress = (i / samples.length) * 100;
                progressCallback(progress);
            }
            
            // 避免阻塞
            if (i % 20 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        if (progressCallback) progressCallback(100);
        return results;
    }

    async predictBatch(samples) {
        const sequences = samples.map(s => s.sequence || '');
        const encodedSequences = sequences.map(seq => this.encodeSequence(seq));
        const oneHotSequences = this.oneHotEncodeBatch(encodedSequences);
        
        const sequenceTensor = tf.tensor3d(oneHotSequences);
        
        try {
            const prediction = this.model.predict(sequenceTensor);
            const values = await prediction.data();
            
            const results = [];
            for (let i = 0; i < samples.length; i++) {
                const startIdx = i * 3;
                const probabilities = [
                    values[startIdx] || 0,
                    values[startIdx + 1] || 0,
                    values[startIdx + 2] || 0
                ];
                
                // 确保概率和为1
                const sum = probabilities.reduce((a, b) => a + b, 0);
                const normalizedProbabilities = probabilities.map(p => p / sum);
                
                const maxIndex = normalizedProbabilities.indexOf(Math.max(...normalizedProbabilities));
                const predictedRisk = ['High', 'Medium', 'Low'][maxIndex];
                
                results.push({
                    predictedRisk: predictedRisk,
                    confidence: normalizedProbabilities[maxIndex],
                    probabilities: {
                        High: normalizedProbabilities[0],
                        Medium: normalizedProbabilities[1],
                        Low: normalizedProbabilities[2]
                    }
                });
            }
            
            return results;
        } finally {
            sequenceTensor.dispose();
        }
    }

    getModelInfo() {
        return {
            version: '3.0-Simple-GRU',
            type: 'Simple GRU Neural Network',
            architecture: 'GRU(32) -> Dense(16) -> Output(3)'
        };
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.isTrained = false;
    }
}

const diseaseModel = new SimpleGRUModel();
