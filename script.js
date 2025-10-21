class DNAClassifier {
    constructor() {
        this.model = null;
        this.trainData = null;
        this.testData = null;
        this.isTraining = false;
        this.classLabels = ['Human', 'Bacteria', 'Virus', 'Plant'];
        this.featureNames = [
            'GC_Content', 'AT_Content', 'Sequence_Length', 
            'Num_A', 'Num_T', 'Num_C', 'Num_G', 'kmer_3_freq'
        ];
        this.modelType = 'improved_dense'; // 默认模型类型
        this.init();
    }

    // 添加模型选择方法
    setModelType(type) {
        this.modelType = type;
        this.log(`Model type set to: ${type}`);
    }

    async trainModel() {
        if (!this.trainData) {
            this.log('Error: Please upload training data first');
            return;
        }

        if (this.isTraining) {
            this.log('Training in progress, please wait...');
            return;
        }

        // 显示数据统计
        if (this.trainData.analysis) {
            this.log('Data Analysis:');
            this.log(`Total samples: ${this.trainData.analysis.totalSamples}`);
            this.log(`Class distribution: ${JSON.stringify(this.trainData.analysis.classDistribution)}`);
        }

        this.isTraining = true;
        this.log(`Starting model training with ${this.modelType} architecture...`);

        try {
            // 创建改进的模型
            this.model = ModelBuilder.createModel(
                this.trainData.features[0].length, 
                this.classLabels.length,
                this.modelType
            );
            
            const { features, labels } = this.trainData;
            const xs = tf.tensor2d(features);
            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

            // 改进的训练配置
            const epochs = 100; // 增加训练轮数
            const batchSize = Math.min(32, features.length); // 动态批次大小
            const validationSplit = 0.2;

            let bestValAcc = 0;
            let patience = 10; // 早停耐心值
            let patienceCounter = 0;

            await this.model.fit(xs, ys, {
                epochs: epochs,
                batchSize: batchSize,
                validationSplit: validationSplit,
                callbacks: {
                    onEpochEnd: async (epoch, logs) => {
                        const currentValAcc = logs.val_acc;
                        const currentAcc = logs.acc;
                        const currentLoss = logs.loss;
                        const currentValLoss = logs.val_loss;
                        
                        this.log(`Epoch ${epoch + 1}/${epochs} - ` +
                                `Acc: ${(currentAcc * 100).toFixed(2)}%, ` +
                                `Val Acc: ${(currentValAcc * 100).toFixed(2)}%, ` +
                                `Loss: ${currentLoss.toFixed(4)}, ` +
                                `Val Loss: ${currentValLoss.toFixed(4)}`);

                        // 简单的早停机制
                        if (currentValAcc > bestValAcc) {
                            bestValAcc = currentValAcc;
                            patienceCounter = 0;
                        } else {
                            patienceCounter++;
                        }

                        if (patienceCounter >= patience) {
                            this.log(`Early stopping triggered at epoch ${epoch + 1}`);
                            this.model.stopTraining = true;
                        }
                    },
                    onTrainEnd: () => {
                        this.log(`Training completed. Best validation accuracy: ${(bestValAcc * 100).toFixed(2)}%`);
                    }
                }
            });

            this.log('Training completed!');
            this.updateModelInfo();

            // 立即在训练数据上进行评估
            await this.evaluateOnTrainData();

        } catch (error) {
            this.log(`Training error: ${error.message}`);
            console.error('Training error details:', error);
        } finally {
            this.isTraining = false;
            
            // 清理内存
            if (xs) xs.dispose();
            if (ys) ys.dispose();
        }
    }

    async evaluateOnTrainData() {
        if (!this.model || !this.trainData) return;

        this.log('Evaluating on training data...');
        
        const { features, labels } = this.trainData;
        const xs = tf.tensor2d(features);
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

        const evaluation = this.model.evaluate(xs, ys);
        const loss = evaluation[0].dataSync()[0];
        const accuracy = evaluation[1].dataSync()[0];

        this.log(`Training set results - Accuracy: ${(accuracy * 100).toFixed(2)}%, Loss: ${loss.toFixed(4)}`);

        xs.dispose();
        ys.dispose();
        evaluation.forEach(tensor => tensor.dispose());
    }

    // 添加数据增强方法
    augmentData(features, labels, augmentationFactor = 0.1) {
        if (features.length === 0) return { features, labels };
        
        const augmentedFeatures = [...features];
        const augmentedLabels = [...labels];
        
        const numAugment = Math.floor(features.length * augmentationFactor);
        
        for (let i = 0; i < numAugment; i++) {
            const randomIndex = Math.floor(Math.random() * features.length);
            const originalFeature = features[randomIndex];
            
            // 添加小的随机噪声
            const noisyFeature = originalFeature.map(value => {
                return value + (Math.random() - 0.5) * 0.1; // 添加 ±5% 的噪声
            });
            
            augmentedFeatures.push(noisyFeature);
            augmentedLabels.push(labels[randomIndex]);
        }
        
        return { features: augmentedFeatures, labels: augmentedLabels };
    }
}
