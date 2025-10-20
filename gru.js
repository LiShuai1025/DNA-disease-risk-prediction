class EnhancedGRUDiseaseModel {
    constructor() {
        this.model = null;
        this.isTrained = false;
        this.maxSequenceLength = 150; // 增加序列长度
        this.vocabSize = 4;
        this.numOutputs = 3; // 改为3类分类
        this.trainingInProgress = false;
        this.classWeights = null; // 用于处理类别不平衡
    }

    // 更丰富的特征提取
    extractAdvancedFeatures(sequence) {
        const features = {};
        
        // GC含量
        const gcCount = (sequence.match(/[GC]/gi) || []).length;
        features.gcContent = (gcCount / sequence.length) * 100;
        
        // AT含量
        const atCount = (sequence.match(/[AT]/gi) || []).length;
        features.atContent = (atCount / sequence.length) * 100;
        
        // 序列熵（复杂度）
        features.entropy = this.calculateEntropy(sequence);
        
        // 重复模式检测
        features.repeatPatterns = this.detectRepeatPatterns(sequence);
        
        // k-mer频率（2-mer, 3-mer, 4-mer）
        features.kmer2Freq = this.calculateKmerFrequency(sequence, 2);
        features.kmer3Freq = this.calculateKmerFrequency(sequence, 3);
        features.kmer4Freq = this.calculateKmerFrequency(sequence, 4);
        
        return features;
    }

    calculateEntropy(sequence) {
        const freq = {};
        for (let base of sequence) {
            freq[base] = (freq[base] || 0) + 1;
        }
        
        let entropy = 0;
        const total = sequence.length;
        for (let base in freq) {
            const p = freq[base] / total;
            entropy -= p * Math.log2(p);
        }
        
        return entropy;
    }

    calculateKmerFrequency(sequence, k) {
        const kmers = new Map();
        for (let i = 0; i <= sequence.length - k; i++) {
            const kmer = sequence.substring(i, i + k);
            kmers.set(kmer, (kmers.get(kmer) || 0) + 1);
        }
        return kmers.size / (sequence.length - k + 1);
    }

    detectRepeatPatterns(sequence) {
        let repeatScore = 0;
        
        // 检测简单重复
        for (let k = 2; k <= 4; k++) {
            for (let i = 0; i <= sequence.length - k * 3; i++) {
                const pattern = sequence.substring(i, i + k);
                let repeats = 1;
                for (let j = i + k; j <= sequence.length - k; j += k) {
                    if (sequence.substring(j, j + k) === pattern) {
                        repeats++;
                    } else {
                        break;
                    }
                }
                if (repeats >= 3) {
                    repeatScore += repeats;
                }
            }
        }
        
        return repeatScore;
    }

    // 改进的序列编码
    encodeSequenceWithFeatures(sequence, additionalFeatures = {}) {
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
        
        return {
            sequence: encoded,
            features: additionalFeatures
        };
    }

    // 更复杂的模型架构
    buildEnhancedModel() {
        const model = tf.sequential();
        
        // 输入层 - 序列数据
        model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            inputShape: [this.maxSequenceLength, this.vocabSize],
            dropout: 0.3,
            recurrentDropout: 0.2
        }));
        
        // 第二层GRU
        model.add(tf.layers.gru({
            units: 32,
            dropout: 0.2,
            recurrentDropout: 0.1
        }));
        
        // 特征输入层（并行）
        const featureInput = tf.input({shape: [6]}); // 6个额外特征
        
        // 合并序列特征和手工特征
        const sequenceOutput = model.outputs[0];
        const featureDense = tf.layers.dense({
            units: 8,
            activation: 'relu'
        }).apply(featureInput);
        
        const concatenated = tf.layers.concatenate().apply([sequenceOutput, featureDense]);
        
        // 全连接层
        const hidden = tf.layers.dense({
            units: 16,
            activation: 'relu',
            kernelRegularizer: tf.regularizers.l2({l2: 0.01})
        }).apply(concatenated);
        
        const dropout = tf.layers.dropout({rate: 0.3}).apply(hidden);
        
        // 输出层 - 3个类别
        const output = tf.layers.dense({
            units: this.numOutputs,
            activation: 'softmax'
        }).apply(dropout);
        
        const enhancedModel = tf.model({
            inputs: [model.inputs[0], featureInput],
            outputs: output
        });
        
        enhancedModel.compile({
            optimizer: tf.train.adam(0.0005), // 更小的学习率
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        
        this.model = enhancedModel;
        console.log('Enhanced GRU Model built successfully');
        return enhancedModel;
    }

    // 计算类别权重处理不平衡数据
    calculateClassWeights(samples) {
        const riskCounts = {High: 0, Medium: 0, Low: 0};
        samples.forEach(sample => {
            riskCounts[sample.actualRisk]++;
        });
        
        const total = samples.length;
        this.classWeights = {
            0: total / (3 * riskCounts.High),   // High risk
            1: total / (3 * riskCounts.Medium), // Medium risk  
            2: total / (3 * riskCounts.Low)     // Low risk
        };
        
        console.log('Class weights:', this.classWeights);
    }

    // 改进的训练方法
    async trainEnhancedModel(samples, progressCallback = null) {
        if (this.trainingInProgress) {
            throw new Error('Training already in progress');
        }
        
        this.trainingInProgress = true;
        
        try {
            if (progressCallback) progressCallback(10, 'Building enhanced model...');
            
            if (this.model) {
                this.model.dispose();
            }
            
            this.buildEnhancedModel();
            this.calculateClassWeights(samples);
            
            if (progressCallback) progressCallback(20, 'Preparing enhanced features...');
            
            // 准备增强的训练数据
            const {sequences, features, labels} = this.prepareEnhancedTrainingData(samples);
            
            if (progressCallback) progressCallback(40, 'Starting enhanced training...');
            
            // 训练更多轮次
            await this.model.fit([sequences, features], labels, {
                epochs: 50,
                batchSize: 16,
                validationSplit: 0.2,
                classWeight: this.classWeights,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        await new Promise(resolve => setTimeout(resolve, 10));
                        
                        const progress = 40 + (epoch / 50) * 50;
                        if (progressCallback) {
                            progressCallback(
                                Math.min(90, progress), 
                                `Epoch ${epoch + 1}/50 - Loss: ${logs.loss.toFixed(4)}, Acc: ${logs.acc.toFixed(4)}`
                            );
                        }
                    }
                }
            });
            
            if (progressCallback) progressCallback(100, 'Enhanced training completed');
            
            this.isTrained = true;
            console.log('Enhanced GRU Model training completed');
            
            // 清理张量
            sequences.dispose();
            features.dispose();
            labels.dispose();
            
        } catch (error) {
            console.error('Error training enhanced model:', error);
            if (this.model) {
                this.model.dispose();
                this.model = null;
            }
            throw error;
        } finally {
            this.trainingInProgress = false;
        }
    }

    prepareEnhancedTrainingData(samples) {
        const sequences = [];
        const featureArray = [];
        const labels = [];
        
        const maxSamples = Math.min(samples.length, 300);
        
        for (let i = 0; i < maxSamples; i++) {
            const sample = samples[i];
            if (sample.sequence && sample.actualRisk) {
                // 提取高级特征
                const advancedFeatures = this.extractAdvancedFeatures(sample.sequence);
                
                // 编码序列
                const encoded = this.encodeSequenceWithFeatures(sample.sequence, advancedFeatures);
                sequences.push(encoded.sequence);
                
                // 添加特征向量
                featureArray.push([
                    advancedFeatures.gcContent / 100, // 归一化
                    advancedFeatures.atContent / 100,
                    advancedFeatures.entropy / 4,     // 近似归一化
                    advancedFeatures.repeatPatterns / 10,
                    advancedFeatures.kmer2Freq,
                    advancedFeatures.kmer3Freq
                ]);
                
                // 3类标签
                const labelIndex = {'High': 0, 'Medium': 1, 'Low': 2}[sample.actualRisk];
                const oneHot = new Array(3).fill(0);
                oneHot[labelIndex] = 1;
                labels.push(oneHot);
            }
        }
        
        console.log(`Prepared ${sequences.length} enhanced samples for training`);
        
        return {
            sequences: tf.tensor3d(this.oneHotEncodeBatch(sequences)),
            features: tf.tensor2d(featureArray),
            labels: tf.tensor2d(labels)
        };
    }

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

    // 改进的预测方法
    async predictEnhancedSamples(samples, progressCallback = null) {
        if (!this.isTrained) {
            throw new Error('Model not trained');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            for (let i = 0; i < totalSamples; i++) {
                const sample = samples[i];
                const result = await this.predictEnhancedSingleSample(sample);
                results.push(result);
                
                if (progressCallback) {
                    const progress = ((i + 1) / totalSamples) * 100;
                    progressCallback(Math.round(progress));
                }
                
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in enhanced prediction:', error);
            throw error;
        }
    }

    async predictEnhancedSingleSample(sample) {
        if (!this.model) {
            return this.getDefaultPrediction();
        }

        try {
            // 提取特征
            const advancedFeatures = this.extractAdvancedFeatures(sample.sequence || '');
            const encoded = this.encodeSequenceWithFeatures(sample.sequence || '', advancedFeatures);
            const oneHot = this.oneHotEncodeBatch([encoded.sequence]);
            
            const sequenceTensor = tf.tensor3d(oneHot);
            const featureTensor = tf.tensor2d([[
                advancedFeatures.gcContent / 100,
                advancedFeatures.atContent / 100,
                advancedFeatures.entropy / 4,
                advancedFeatures.repeatPatterns / 10,
                advancedFeatures.kmer2Freq,
                advancedFeatures.kmer3Freq
            ]]);
            
            // 预测
            const prediction = this.model.predict([sequenceTensor, featureTensor]);
            const values = await prediction.data();
            
            // 解析结果
            const probabilities = Array.from(values);
            const maxIndex = probabilities.indexOf(Math.max(...probabilities));
            const predictedRisk = ['High', 'Medium', 'Low'][maxIndex];
            const confidence = probabilities[maxIndex];
            
            // 清理
            sequenceTensor.dispose();
            featureTensor.dispose();
            prediction.dispose();
            
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
            console.error('Error predicting enhanced sample:', error);
            return this.getDefaultPrediction();
        }
    }

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

    getModelInfo() {
        return {
            version: '4.0-GRU-Enhanced',
            type: 'Enhanced Multi-Output GRU Neural Network',
            architecture: 'GRU(64)->GRU(32) + FeatureDense(8) -> Dense(16) -> Output(3)',
            outputs: ['High', 'Medium', 'Low'],
            features: 'GC Content, AT Content, Entropy, Repeat Patterns, k-mer frequencies'
        };
    }
}

// 替换原来的模型
const diseaseModel = new EnhancedGRUDiseaseModel();
