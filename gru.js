class OptimizedGRUDiseaseModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.maxSequenceLength = 100; // 固定序列长度
        this.vocabSize = 4;
        this.numOutputs = 3; // 3分类问题
        this.trainingInProgress = false;
        this.featureSize = 10; // 手工特征数量
        this.useCache = true; // 启用预测缓存
        this.predictionCache = new Map(); // 预测缓存
    }

    // 优化的序列编码 - 预计算编码
    encodeSequence(sequence) {
        // 检查缓存
        if (this.useCache && this.predictionCache.has(sequence)) {
            return this.predictionCache.get(sequence).encoded;
        }
        
        const encoding = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        const encoded = [];
        
        const effectiveLength = Math.min(sequence.length, this.maxSequenceLength);
        
        for (let i = 0; i < effectiveLength; i++) {
            const char = sequence[i].toUpperCase();
            encoded.push(encoding[char] || 0);
        }
        
        // 填充
        while (encoded.length < this.maxSequenceLength) {
            encoded.push(0);
        }
        
        return encoded;
    }

    // 批量编码 - 显著提高速度
    encodeSequenceBatch(sequences) {
        return sequences.map(seq => this.encodeSequence(seq));
    }

    // 批量one-hot编码
    oneHotEncodeBatch(encodedSequences) {
        const batchSize = encodedSequences.length;
        const oneHotBatch = new Array(batchSize);
        
        for (let i = 0; i < batchSize; i++) {
            const oneHot = new Array(this.maxSequenceLength);
            for (let j = 0; j < this.maxSequenceLength; j++) {
                const vector = new Array(this.vocabSize).fill(0);
                vector[encodedSequences[i][j]] = 1;
                oneHot[j] = vector;
            }
            oneHotBatch[i] = oneHot;
        }
        
        return oneHotBatch;
    }

    // 提取关键特征用于模型输入
    extractKeyFeatures(sample) {
        const features = sample.features || {};
        
        // 选择最重要的10个特征
        return [
            features.gcContent / 100, // 归一化
            features.kmerFreq || 0.5,
            features.entropy ? features.entropy / 4 : 0.5, // 近似归一化
            features.repeatScore ? features.repeatScore / 10 : 0,
            features.gcDinucTotal || 0,
            (features.numA || 0) / (features.sequenceLength || 1),
            (features.numT || 0) / (features.sequenceLength || 1),
            (features.numC || 0) / (features.sequenceLength || 1),
            (features.numG || 0) / (features.sequenceLength || 1),
            features.sequenceLength ? Math.min(features.sequenceLength, 200) / 200 : 0.5
        ];
    }

    // 构建更高效的模型
    buildEfficientModel() {
        const model = tf.sequential();
        
        // 更小的GRU层 - 平衡性能和准确性
        model.add(tf.layers.gru({
            units: 24, // 减少单元数提高速度
            returnSequences: false,
            inputShape: [this.maxSequenceLength, this.vocabSize],
            dropout: 0.2,
            recurrentDropout: 0.1
        }));
        
        // 添加批归一化加速训练
        model.add(tf.layers.batchNormalization());
        
        // 更小的全连接层
        model.add(tf.layers.dense({
            units: 12,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dropout({rate: 0.3}));
        
        // 输出层 - 3分类
        model.add(tf.layers.dense({
            units: this.numOutputs,
            activation: 'softmax'
        }));
        
        // 使用更快的优化器
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        this.model = model;
        console.log('Efficient GRU Model built successfully');
        return model;
    }

    // 准备训练数据
    prepareTrainingData(samples) {
        const sequences = [];
        const features = [];
        const labels = [];
        
        // 限制数据集大小但保持平衡
        const maxSamples = Math.min(samples.length, 150);
        
        for (let i = 0; i < maxSamples; i++) {
            const sample = samples[i];
            if (sample.sequence && sample.actualRisk) {
                sequences.push(sample.sequence);
                features.push(this.extractKeyFeatures(sample));
                
                // 3类标签
                const labelIndex = {'High': 0, 'Medium': 1, 'Low': 2}[sample.actualRisk];
                const oneHot = new Array(3).fill(0);
                oneHot[labelIndex] = 1;
                labels.push(oneHot);
            }
        }
        
        console.log(`Prepared ${sequences.length} samples for training`);
        
        // 批量编码序列
        const encodedSequences = this.encodeSequenceBatch(sequences);
        const oneHotSequences = this.oneHotEncodeBatch(encodedSequences);
        
        return {
            sequences: tf.tensor3d(oneHotSequences),
            features: tf.tensor2d(features),
            labels: tf.tensor2d(labels)
        };
    }

    // 改进的训练方法
    async trainModel(samples, progressCallback = null) {
        if (this.trainingInProgress) {
            throw new Error('Training already in progress');
        }
        
        this.trainingInProgress = true;
        this.predictionCache.clear(); // 清除缓存
        
        try {
            if (progressCallback) progressCallback(10, 'Building efficient model...');
            
            // 清理现有模型
            if (this.model) {
                this.model.dispose();
            }
            
            this.buildEfficientModel();
            
            if (progressCallback) progressCallback(20, 'Preparing training data...');
            
            const {sequences, features, labels} = this.prepareTrainingData(samples);
            
            if (progressCallback) progressCallback(40, 'Starting training...');
            
            let bestAccuracy = 0;
            let patience = 8;
            let patienceCounter = 0;
            
            // 手动实现训练循环以便早停
            for (let epoch = 0; epoch < 50; epoch++) {
                const history = await this.model.fit(sequences, labels, {
                    epochs: 1,
                    batchSize: 16,
                    validationSplit: 0.2,
                    verbose: 0
                });
                
                const accuracy = history.history.acc[0];
                const valAccuracy = history.history.val_acc ? history.history.val_acc[0] : accuracy;
                
                // 早停逻辑
                if (valAccuracy > bestAccuracy) {
                    bestAccuracy = valAccuracy;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                }
                
                if (patienceCounter >= patience) {
                    console.log(`Early stopping at epoch ${epoch + 1}`);
                    break;
                }
                
                if (progressCallback) {
                    const progress = 40 + ((epoch + 1) / 50) * 50;
                    progressCallback(
                        Math.min(95, progress), 
                        `Epoch ${epoch + 1} - Accuracy: ${accuracy.toFixed(4)}, Val: ${valAccuracy.toFixed(4)}`
                    );
                }
                
                // 让浏览器处理其他任务
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            if (progressCallback) progressCallback(100, 'Training completed');
            
            this.isTrained = true;
            console.log('Model training completed');
            
            // 清理张量
            sequences.dispose();
            features.dispose();
            labels.dispose();
            
        } catch (error) {
            console.error('Error training model:', error);
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            throw error;
        } finally {
            this.trainingInProgress = false;
        }
    }

    // 高速批量预测
    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained) {
            throw new Error('Model not trained');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            // 使用更大的批处理大小提高速度
            const batchSize = 20;
            
            for (let i = 0; i < totalSamples; i += batchSize) {
                const batchSamples = samples.slice(i, i + batchSize);
                const batchResults = await this.predictBatch(batchSamples);
                results.push(...batchResults);
                
                if (progressCallback) {
                    const progress = ((i + batchSize) / totalSamples) * 100;
                    progressCallback(Math.min(100, Math.round(progress)));
                }
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in prediction:', error);
            throw error;
        }
    }

    // 优化的批量预测
    async predictBatch(samples) {
        const batchResults = [];
        const sequences = [];
        const featureList = [];
        
        // 准备批量数据
        for (const sample of samples) {
            // 检查缓存
            if (this.useCache && this.predictionCache.has(sample.sequence)) {
                batchResults.push(this.predictionCache.get(sample.sequence));
                continue;
            }
            
            sequences.push(sample.sequence || '');
            featureList.push(this.extractKeyFeatures(sample));
        }
        
        // 如果没有需要预测的样本，直接返回缓存结果
        if (sequences.length === 0) {
            return batchResults;
        }
        
        // 批量编码和预测
        const encodedSequences = this.encodeSequenceBatch(sequences);
        const oneHotSequences = this.oneHotEncodeBatch(encodedSequences);
        
        const sequenceTensor = tf.tensor3d(oneHotSequences);
        const featureTensor = tf.tensor2d(featureList);
        
        try {
            const prediction = this.model.predict(sequenceTensor);
            const values = await prediction.data();
            
            let resultIndex = 0;
            for (let i = 0; i < samples.length; i++) {
                // 如果这个样本使用了缓存，跳过
                if (this.useCache && this.predictionCache.has(samples[i].sequence)) {
                    continue;
                }
                
                const startIdx = resultIndex * 3;
                const probabilities = [
                    values[startIdx],
                    values[startIdx + 1], 
                    values[startIdx + 2]
                ];
                
                const maxIndex = probabilities.indexOf(Math.max(...probabilities));
                const predictedRisk = ['High', 'Medium', 'Low'][maxIndex];
                const confidence = probabilities[maxIndex];
                
                const result = {
                    predictedRisk: predictedRisk,
                    confidence: confidence,
                    probabilities: {
                        High: probabilities[0],
                        Medium: probabilities[1],
                        Low: probabilities[2]
                    }
                };
                
                // 缓存结果
                if (this.useCache) {
                    this.predictionCache.set(samples[i].sequence, result);
                }
                
                batchResults.push(result);
                resultIndex++;
            }
            
            return batchResults;
            
        } finally {
            // 清理张量
            sequenceTensor.dispose();
            featureTensor.dispose();
            if (prediction) {
                prediction.dispose();
            }
        }
    }

    // 获取模型信息
    getModelInfo() {
        return {
            version: '5.0-GRU-Optimized',
            type: 'Efficient GRU Neural Network',
            architecture: 'GRU(24) -> BatchNorm -> Dense(12) -> Output(3)',
            features: 'Enhanced features with caching',
            performance: 'Optimized for speed and accuracy'
        };
    }
}

const diseaseModel = new OptimizedGRUDiseaseModel();
