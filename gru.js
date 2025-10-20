class StableGRUDiseaseModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.maxSequenceLength = 100;
        this.vocabSize = 4;
        this.numOutputs = 3;
        this.trainingInProgress = false;
        this.predictionCache = new Map();
    }

    // 稳定的序列编码
    encodeSequence(sequence) {
        if (!sequence || typeof sequence !== 'string') {
            return new Array(this.maxSequenceLength).fill(0);
        }

        const encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        const encoded = [];
        
        const effectiveSequence = sequence.toUpperCase();
        const effectiveLength = Math.min(effectiveSequence.length, this.maxSequenceLength);
        
        for (let i = 0; i < effectiveLength; i++) {
            const char = effectiveSequence[i];
            encoded.push(encoding[char] || 0);
        }
        
        // 填充
        while (encoded.length < this.maxSequenceLength) {
            encoded.push(0);
        }
        
        return encoded;
    }

    // 稳定的批量编码
    oneHotEncodeBatch(encodedSequences) {
        const batchSize = encodedSequences.length;
        const oneHotBatch = [];
        
        for (let i = 0; i < batchSize; i++) {
            const oneHot = [];
            for (let j = 0; j < this.maxSequenceLength; j++) {
                const vector = new Array(this.vocabSize).fill(0);
                const baseIndex = encodedSequences[i][j];
                if (baseIndex >= 0 && baseIndex < this.vocabSize) {
                    vector[baseIndex] = 1;
                }
                oneHot.push(vector);
            }
            oneHotBatch.push(oneHot);
        }
        
        return oneHotBatch;
    }

    // 提取关键特征
    extractKeyFeatures(sample) {
        const features = sample.features || {};
        
        return [
            (features.gcContent || 50) / 100,
            features.kmerFreq || 0.5,
            (features.entropy || 2) / 4,
            (features.repeatScore || 0) / 10,
            features.gcDinucTotal || 0,
            (features.numA || 25) / 100,
            (features.numT || 25) / 100,
            (features.numC || 25) / 100,
            (features.numG || 25) / 100,
            Math.min(features.sequenceLength || 100, 200) / 200
        ];
    }

    // 构建稳定的模型
    buildStableModel() {
        try {
            const model = tf.sequential();
            
            model.add(tf.layers.gru({
                units: 32,
                returnSequences: false,
                inputShape: [this.maxSequenceLength, this.vocabSize],
                dropout: 0.2,
                recurrentDropout: 0.1
            }));
            
            model.add(tf.layers.dense({
                units: 16,
                activation: 'relu'
            }));
            
            model.add(tf.layers.dropout({rate: 0.3}));
            
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
            console.log('Stable GRU Model built successfully');
            return model;
        } catch (error) {
            console.error('Error building model:', error);
            throw error;
        }
    }

    // 准备训练数据
    prepareTrainingData(samples) {
        const sequences = [];
        const labels = [];
        
        const maxSamples = Math.min(samples.length, 150);
        
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
        
        console.log(`Prepared ${sequences.length} samples for training`);
        
        const encodedSequences = sequences.map(seq => this.encodeSequence(seq));
        const oneHotSequences = this.oneHotEncodeBatch(encodedSequences);
        
        return {
            sequences: tf.tensor3d(oneHotSequences),
            labels: tf.tensor2d(labels)
        };
    }

    // 稳定的训练方法
    async trainModel(samples, progressCallback = null) {
        if (this.trainingInProgress) {
            throw new Error('Training already in progress');
        }
        
        this.trainingInProgress = true;
        this.predictionCache.clear();
        
        try {
            if (progressCallback) progressCallback(10, 'Building model...');
            
            // 清理现有模型
            if (this.model) {
                this.model.dispose();
            }
            
            this.buildStableModel();
            
            if (progressCallback) progressCallback(30, 'Preparing data...');
            
            const {sequences, labels} = this.prepareTrainingData(samples);
            
            if (progressCallback) progressCallback(50, 'Starting training...');
            
            // 简化训练过程
            await this.model.fit(sequences, labels, {
                epochs: 25,
                batchSize: 16,
                validationSplit: 0.2,
                verbose: 0,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        const progress = 50 + ((epoch + 1) / 25) * 45;
                        if (progressCallback) {
                            const message = `Epoch ${epoch + 1}/25 - Loss: ${logs.loss.toFixed(4)}`;
                            progressCallback(Math.min(95, progress), message);
                        }
                        await new Promise(resolve => setTimeout(resolve, 10));
                    }
                }
            });
            
            if (progressCallback) progressCallback(100, 'Training completed');
            
            this.isTrained = true;
            console.log('Model training completed successfully');
            
            // 清理张量
            sequences.dispose();
            labels.dispose();
            
        } catch (error) {
            console.error('Training error:', error);
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            throw error;
        } finally {
            this.trainingInProgress = false;
        }
    }

    // 稳定的预测方法 - 使用逐样本预测避免复杂错误
    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained || !this.model) {
            throw new Error('Model not trained or available');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            for (let i = 0; i < totalSamples; i++) {
                const sample = samples[i];
                let result;
                
                // 检查缓存
                if (this.predictionCache.has(sample.sequence)) {
                    result = this.predictionCache.get(sample.sequence);
                } else {
                    result = await this.predictSingleSampleStable(sample);
                    this.predictionCache.set(sample.sequence, result);
                }
                
                results.push(result);
                
                if (progressCallback && (i % 5 === 0 || i === totalSamples - 1)) {
                    const progress = ((i + 1) / totalSamples) * 100;
                    progressCallback(Math.round(progress));
                }
                
                // 定期让出控制权
                if (i % 10 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 0));
                }
            }
            
            return results;
            
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    // 完全重写的单样本预测 - 确保稳定性
    async predictSingleSampleStable(sample) {
        if (!this.model) {
            return this.getDefaultPrediction();
        }

        let sequenceTensor;
        let predictionTensor;

        try {
            // 编码序列
            const encoded = this.encodeSequence(sample.sequence);
            const oneHot = this.oneHotEncodeBatch([encoded]);
            sequenceTensor = tf.tensor3d(oneHot);
            
            // 进行预测
            predictionTensor = this.model.predict(sequenceTensor);
            const values = await predictionTensor.data();
            
            // 处理结果
            const probabilities = Array.from(values);
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            const predictedRisk = ['High', 'Medium', 'Low'][maxIndex];
            const confidence = probabilities[maxIndex];
            
            return {
                predictedRisk: predictedRisk,
                confidence: confidence,
                probabilities: {
                    High: probabilities[0],
                    Medium: probabilities[1],
                    Low: probabilities[2]
                }
            };
            
        } catch (error) {
            console.error('Single sample prediction error:', error);
            return this.getDefaultPrediction();
        } finally {
            // 确保清理所有张量
            if (sequenceTensor) {
                sequenceTensor.dispose();
            }
            if (predictionTensor) {
                predictionTensor.dispose();
            }
        }
    }

    // 默认预测结果
    getDefaultPrediction() {
        return {
            predictedRisk: 'Medium',
            confidence: 0.33,
            probabilities: {
                High: 0.33,
                Medium: 0.34,
                Low: 0.33
            }
        };
    }

    // 获取模型信息
    getModelInfo() {
        return {
            version: '6.0-GRU-Stable',
            type: 'Stable GRU Neural Network',
            architecture: 'GRU(32) -> Dense(16) -> Output(3)',
            features: 'Stable prediction with error handling'
        };
    }

    // 清理资源
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
        }
        this.predictionCache.clear();
        this.isTrained = false;
    }
}

const diseaseModel = new StableGRUDiseaseModel();
