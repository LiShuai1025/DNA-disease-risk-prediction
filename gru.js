class RobustGRUDiseaseModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.maxSequenceLength = 100;
        this.vocabSize = 4;
        this.numOutputs = 3;
        this.trainingInProgress = false;
        this.predictionCache = new Map();
        this.trainingHistory = [];
    }

    // 序列编码
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

    // 批量编码
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

    // 构建更健壮的模型
    buildRobustModel() {
        try {
            const model = tf.sequential();
            
            // 第一层GRU
            model.add(tf.layers.gru({
                units: 48,
                returnSequences: true,
                inputShape: [this.maxSequenceLength, this.vocabSize],
                dropout: 0.3,
                recurrentDropout: 0.2
            }));
            
            // 第二层GRU
            model.add(tf.layers.gru({
                units: 24,
                dropout: 0.2,
                recurrentDropout: 0.1
            }));
            
            // 批归一化
            model.add(tf.layers.batchNormalization());
            
            // 全连接层
            model.add(tf.layers.dense({
                units: 16,
                activation: 'relu',
                kernelRegularizer: tf.regularizers.l2({l2: 0.01})
            }));
            
            model.add(tf.layers.dropout({rate: 0.4}));
            
            // 输出层
            model.add(tf.layers.dense({
                units: this.numOutputs,
                activation: 'softmax'
            }));
            
            // 使用更稳定的优化器配置
            model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy']
            });
            
            this.model = model;
            console.log('Robust GRU Model built successfully');
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
        
        // 使用更多样本进行训练
        const maxSamples = Math.min(samples.length, 250);
        
        console.log(`Using ${maxSamples} samples for training`);
        
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

    // 改进的训练方法 - 添加更多监控和验证
    async trainModel(samples, progressCallback = null) {
        if (this.trainingInProgress) {
            throw new Error('Training already in progress');
        }
        
        this.trainingInProgress = true;
        this.predictionCache.clear();
        this.trainingHistory = [];
        
        let sequencesTensor, labelsTensor;
        
        try {
            if (progressCallback) progressCallback(10, 'Building robust model...');
            
            // 清理现有模型
            if (this.model) {
                this.model.dispose();
            }
            
            this.buildRobustModel();
            
            if (progressCallback) progressCallback(20, 'Preparing training data...');
            
            const trainingData = this.prepareTrainingData(samples);
            sequencesTensor = trainingData.sequences;
            labelsTensor = trainingData.labels;
            
            if (progressCallback) progressCallback(30, 'Starting training process...');
            
            // 手动实现训练循环以便更好的控制
            const epochs = 40;
            const batchSize = 16;
            const validationSplit = 0.2;
            
            // 计算训练和验证样本数量
            const numSamples = sequencesTensor.shape[0];
            const numTrainSamples = Math.floor(numSamples * (1 - validationSplit));
            
            // 分割数据
            const trainSequences = sequencesTensor.slice([0, 0, 0], [numTrainSamples, this.maxSequenceLength, this.vocabSize]);
            const trainLabels = labelsTensor.slice([0, 0], [numTrainSamples, this.numOutputs]);
            const valSequences = sequencesTensor.slice([numTrainSamples, 0, 0], [numSamples - numTrainSamples, this.maxSequenceLength, this.vocabSize]);
            const valLabels = labelsTensor.slice([numTrainSamples, 0], [numSamples - numTrainSamples, this.numOutputs]);
            
            let bestValAccuracy = 0;
            let patience = 10;
            let patienceCounter = 0;
            
            for (let epoch = 0; epoch < epochs; epoch++) {
                if (progressCallback) {
                    const progress = 30 + ((epoch + 1) / epochs) * 60;
                    progressCallback(Math.min(90, progress), `Epoch ${epoch + 1}/${epochs} - Training...`);
                }
                
                // 训练一个epoch
                const history = await this.model.fit(trainSequences, trainLabels, {
                    epochs: 1,
                    batchSize: batchSize,
                    verbose: 0
                });
                
                // 验证
                const valResult = await this.model.evaluate(valSequences, valLabels);
                const valAccuracy = valResult[1].dataSync()[0];
                const trainAccuracy = history.history.acc[0];
                const trainLoss = history.history.loss[0];
                
                // 记录训练历史
                this.trainingHistory.push({
                    epoch: epoch + 1,
                    trainAccuracy,
                    trainLoss,
                    valAccuracy
                });
                
                console.log(`Epoch ${epoch + 1}/${epochs} - Train Acc: ${trainAccuracy.toFixed(4)}, Val Acc: ${valAccuracy.toFixed(4)}, Loss: ${trainLoss.toFixed(4)}`);
                
                // 早停逻辑
                if (valAccuracy > bestValAccuracy) {
                    bestValAccuracy = valAccuracy;
                    patienceCounter = 0;
                } else {
                    patienceCounter++;
                }
                
                if (patienceCounter >= patience) {
                    console.log(`Early stopping at epoch ${epoch + 1}, best val accuracy: ${bestValAccuracy.toFixed(4)}`);
                    break;
                }
                
                // 更新进度
                if (progressCallback) {
                    const progress = 30 + ((epoch + 1) / epochs) * 60;
                    const message = `Epoch ${epoch + 1}/${epochs} - Train: ${trainAccuracy.toFixed(4)}, Val: ${valAccuracy.toFixed(4)}`;
                    progressCallback(Math.min(90, progress), message);
                }
                
                // 让浏览器处理其他任务
                await new Promise(resolve => setTimeout(resolve, 50));
            }
            
            if (progressCallback) progressCallback(100, 'Training completed');
            
            this.isTrained = true;
            console.log('Model training completed successfully');
            console.log('Training history:', this.trainingHistory);
            
        } catch (error) {
            console.error('Training error:', error);
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            throw error;
        } finally {
            // 清理张量
            if (sequencesTensor) sequencesTensor.dispose();
            if (labelsTensor) labelsTensor.dispose();
            this.trainingInProgress = false;
        }
    }

    // 稳定的预测方法
    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained || !this.model) {
            throw new Error('Model not trained or available');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            // 使用适中的批处理大小
            const batchSize = 16;
            
            for (let i = 0; i < totalSamples; i += batchSize) {
                const batchSamples = samples.slice(i, i + batchSize);
                const batchResults = await this.predictBatch(batchSamples);
                results.push(...batchResults);
                
                if (progressCallback) {
                    const progress = ((i + batchSize) / totalSamples) * 100;
                    progressCallback(Math.min(100, Math.round(progress)));
                }
                
                // 定期让出控制权
                if (i % 50 === 0) {
                    await new Promise(resolve => setTimeout(resolve, 10));
                }
            }
            
            return results;
            
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    // 批量预测
    async predictBatch(samples) {
        const batchResults = [];
        const sequences = [];
        
        // 准备批量数据
        for (const sample of samples) {
            // 检查缓存
            if (this.predictionCache.has(sample.sequence)) {
                batchResults.push(this.predictionCache.get(sample.sequence));
                continue;
            }
            
            sequences.push(sample.sequence || '');
        }
        
        // 如果没有需要预测的样本，直接返回缓存结果
        if (sequences.length === 0) {
            return batchResults;
        }
        
        // 批量编码和预测
        const encodedSequences = this.encodeSequenceBatch(sequences);
        const oneHotSequences = this.oneHotEncodeBatch(encodedSequences);
        
        const sequenceTensor = tf.tensor3d(oneHotSequences);
        
        try {
            const prediction = this.model.predict(sequenceTensor);
            const values = await prediction.data();
            
            let resultIndex = 0;
            for (let i = 0; i < samples.length; i++) {
                // 如果这个样本使用了缓存，跳过
                if (this.predictionCache.has(samples[i].sequence)) {
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
                this.predictionCache.set(samples[i].sequence, result);
                
                batchResults.push(result);
                resultIndex++;
            }
            
            return batchResults;
            
        } finally {
            // 清理张量
            sequenceTensor.dispose();
            if (prediction) {
                prediction.dispose();
            }
        }
    }

    // 批量编码序列
    encodeSequenceBatch(sequences) {
        return sequences.map(seq => this.encodeSequence(seq));
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
        const lastEpoch = this.trainingHistory.length > 0 ? 
            this.trainingHistory[this.trainingHistory.length - 1] : null;
        
        const bestEpoch = this.trainingHistory.reduce((best, current) => {
            return current.valAccuracy > best.valAccuracy ? current : best;
        }, {valAccuracy: 0});
        
        return {
            version: '7.0-GRU-Robust',
            type: 'Robust GRU Neural Network',
            architecture: 'GRU(48)->GRU(24) -> BatchNorm -> Dense(16) -> Output(3)',
            trainingEpochs: this.trainingHistory.length,
            bestValAccuracy: bestEpoch.valAccuracy ? (bestEpoch.valAccuracy * 100).toFixed(2) + '%' : 'N/A',
            finalValAccuracy: lastEpoch ? (lastEpoch.valAccuracy * 100).toFixed(2) + '%' : 'N/A'
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
        this.trainingHistory = [];
    }
}

const diseaseModel = new RobustGRUDiseaseModel();
