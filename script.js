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
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.log('系统初始化完成');
    }

    setupEventListeners() {
        // 文件上传监听
        document.getElementById('trainFile').addEventListener('change', (e) => this.handleFileUpload(e, 'train'));
        document.getElementById('testFile').addEventListener('change', (e) => this.handleFileUpload(e, 'test'));
        document.getElementById('modelJsonFile').addEventListener('change', (e) => this.updateFileName(e, 'modelJson'));
        document.getElementById('modelWeightsFile').addEventListener('change', (e) => this.updateFileName(e, 'modelWeights'));

        // 按钮监听
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModel());
        document.getElementById('evaluateBtn').addEventListener('click', () => this.evaluateModel());
        document.getElementById('testRandomBtn').addEventListener('click', () => this.testRandomSamples());
        document.getElementById('saveModelBtn').addEventListener('click', () => this.saveModel());
        document.getElementById('resetBtn').addEventListener('click', () => this.resetSystem());
        document.getElementById('loadModelBtn').addEventListener('click', () => this.loadModel());
        document.getElementById('testSingleBtn').addEventListener('click', () => this.testSingleSequence());
    }

    log(message) {
        const logContainer = document.getElementById('logContainer');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        logEntry.innerHTML = `<strong>[${timestamp}]</strong> ${message}`;
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    updateFileName(event, type) {
        const file = event.target.files[0];
        const fileNameElements = {
            'train': 'trainFileName',
            'test': 'testFileName',
            'modelJson': 'modelJsonFileName',
            'modelWeights': 'modelWeightsFileName'
        };
        
        if (file) {
            document.getElementById(fileNameElements[type]).textContent = file.name;
        }
    }

    async handleFileUpload(event, dataType) {
        const file = event.target.files[0];
        if (!file) return;

        this.log(`加载${dataType === 'train' ? '训练' : '测试'}数据: ${file.name}`);
        
        try {
            const data = await DataLoader.loadCSV(file);
            
            if (dataType === 'train') {
                this.trainData = DataLoader.processData(data);
                document.getElementById('trainSamples').textContent = this.trainData.features.length;
                this.log(`训练数据加载完成: ${this.trainData.features.length} 个样本`);
            } else {
                this.testData = DataLoader.processData(data);
                document.getElementById('testSamples').textContent = this.testData.features.length;
                this.log(`测试数据加载完成: ${this.testData.features.length} 个样本`);
            }
            
            this.updateFileName(event, dataType);
        } catch (error) {
            this.log(`错误: 加载数据失败 - ${error.message}`);
        }
    }

    async trainModel() {
        if (!this.trainData) {
            this.log('错误: 请先上传训练数据');
            return;
        }

        if (this.isTraining) {
            this.log('训练正在进行中，请等待...');
            return;
        }

        this.isTraining = true;
        this.log('开始训练模型...');

        try {
            // 创建或重置模型
            this.model = ModelBuilder.createModel(this.trainData.features[0].length, this.classLabels.length);
            
            const { features, labels } = this.trainData;
            const xs = tf.tensor2d(features);
            const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

            // 训练配置
            const epochs = 50;
            const batchSize = 32;
            const validationSplit = 0.2;

            await this.model.fit(xs, ys, {
                epochs: epochs,
                batchSize: batchSize,
                validationSplit: validationSplit,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        this.log(`Epoch ${epoch + 1}/${epochs} - 准确率: ${(logs.acc * 100).toFixed(2)}%, 验证准确率: ${(logs.val_acc * 100).toFixed(2)}%, 损失: ${logs.loss.toFixed(4)}`);
                    }
                }
            });

            this.log('训练完成!');
            this.updateModelInfo();

        } catch (error) {
            this.log(`训练错误: ${error.message}`);
        } finally {
            this.isTraining = false;
        }
    }

    async evaluateModel() {
        if (!this.model) {
            this.log('错误: 没有可用的模型，请先训练或加载模型');
            return;
        }

        if (!this.testData) {
            this.log('错误: 请先上传测试数据');
            return;
        }

        this.log('开始评估模型...');

        const { features, labels } = this.testData;
        const xs = tf.tensor2d(features);
        const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), this.classLabels.length);

        const evaluation = this.model.evaluate(xs, ys);
        const loss = evaluation[0].dataSync()[0];
        const accuracy = evaluation[1].dataSync()[0];

        this.log(`评估结果 - 准确率: ${(accuracy * 100).toFixed(2)}%, 损失: ${loss.toFixed(4)}`);

        xs.dispose();
        ys.dispose();
        evaluation.forEach(tensor => tensor.dispose());
    }

    async testRandomSamples() {
        if (!this.model || !this.testData) {
            this.log('错误: 没有可用的模型或测试数据');
            return;
        }

        const { features, labels, rawData } = this.testData;
        const resultsContainer = document.getElementById('randomTestResults');
        resultsContainer.innerHTML = '';

        // 随机选择5个样本
        const indices = [];
        for (let i = 0; i < Math.min(5, features.length); i++) {
            indices.push(Math.floor(Math.random() * features.length));
        }

        for (const index of indices) {
            const feature = features[index];
            const trueLabel = labels[index];
            const sequence = rawData[index].Sequence || 'N/A';

            const inputTensor = tf.tensor2d([feature]);
            const prediction = this.model.predict(inputTensor);
            const results = await prediction.data();
            
            const predictedClass = this.classLabels[results.indexOf(Math.max(...results))];
            const trueClass = this.classLabels[trueLabel];
            const confidence = Math.max(...results) * 100;

            const resultDiv = document.createElement('div');
            resultDiv.className = `random-test-item ${predictedClass === trueClass ? '' : 'error'}`;
            resultDiv.innerHTML = `
                <strong>样本 ${index + 1}</strong><br>
                <small>序列: ${sequence.substring(0, 50)}${sequence.length > 50 ? '...' : ''}</small><br>
                真实标签: ${trueClass} | 预测: ${predictedClass}<br>
                置信度: ${confidence.toFixed(2)}%<br>
                ${predictedClass === trueClass ? '✅ 正确' : '❌ 错误'}
            `;

            resultsContainer.appendChild(resultDiv);
            inputTensor.dispose();
            prediction.dispose();
        }

        this.log('随机测试完成');
    }

    async testSingleSequence() {
        const sequenceInput = document.getElementById('singleSequence').value.trim().toUpperCase();
        
        if (!sequenceInput) {
            this.log('错误: 请输入DNA序列');
            return;
        }

        if (!/^[ATCG]+$/.test(sequenceInput)) {
            this.log('错误: DNA序列只能包含 A, T, C, G 字符');
            return;
        }

        if (!this.model) {
            this.log('错误: 没有可用的模型，请先训练或加载模型');
            return;
        }

        try {
            // 提取特征
            const features = DataLoader.extractFeaturesFromSequence(sequenceInput);
            const inputTensor = tf.tensor2d([features]);
            const prediction = this.model.predict(inputTensor);
            const results = await prediction.data();
            
            const maxConfidence = Math.max(...results);
            const predictedClass = this.classLabels[results.indexOf(maxConfidence)];
            
            const resultDiv = document.getElementById('singleTestResult');
            resultDiv.innerHTML = `
                <div class="prediction-result">预测结果: ${predictedClass}</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${maxConfidence * 100}%"></div>
                </div>
                <div>置信度: ${(maxConfidence * 100).toFixed(2)}%</div>
                <div class="confidence-breakdown">
                    ${this.classLabels.map((label, index) => 
                        `${label}: ${(results[index] * 100).toFixed(2)}%`
                    ).join(' | ')}
                </div>
            `;

            this.log(`单序列测试完成: ${predictedClass} (${(maxConfidence * 100).toFixed(2)}%)`);
            
            inputTensor.dispose();
            prediction.dispose();
        } catch (error) {
            this.log(`单序列测试错误: ${error.message}`);
        }
    }

    async saveModel() {
        if (!this.model) {
            this.log('错误: 没有可保存的模型');
            return;
        }

        this.log('保存模型中...');

        try {
            // 保存模型结构
            const modelJson = this.model.toJSON();
            const modelJsonStr = JSON.stringify(modelJson);
            const modelJsonBlob = new Blob([modelJsonStr], { type: 'application/json' });
            
            // 触发下载
            const modelJsonUrl = URL.createObjectURL(modelJsonBlob);
            const modelJsonLink = document.createElement('a');
            modelJsonLink.href = modelJsonUrl;
            modelJsonLink.download = 'dna-model.json';
            modelJsonLink.click();

            // 保存权重
            await this.model.save('downloads://dna-model');

            this.log('模型保存成功! 检查下载文件夹获取 dna-model.json 和 dna-model.weights.bin');
        } catch (error) {
            this.log(`保存模型错误: ${error.message}`);
        }
    }

    async loadModel() {
        const jsonFile = document.getElementById('modelJsonFile').files[0];
        const weightsFile = document.getElementById('modelWeightsFile').files[0];

        if (!jsonFile || !weightsFile) {
            this.log('错误: 请选择模型JSON和权重文件');
            return;
        }

        this.log('加载模型中...');

        try {
            const modelJson = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(JSON.parse(reader.result));
                reader.onerror = reject;
                reader.readAsText(jsonFile);
            });

            const modelWeights = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsArrayBuffer(weightsFile);
            });

            this.model = await tf.loadLayersModel(tf.io.fromMemory(modelJson, modelWeights));
            this.log('模型加载成功!');
            this.updateModelInfo();
        } catch (error) {
            this.log(`加载模型错误: ${error.message}`);
        }
    }

    updateModelInfo() {
        if (!this.model) return;

        let totalParams = 0;
        this.model.summary(null, null, (line) => {
            const match = line.match(/params: (\d+)/);
            if (match) {
                totalParams += parseInt(match[1]);
            }
        });

        document.getElementById('layersCount').textContent = this.model.layers.length;
        document.getElementById('totalParams').textContent = totalParams.toLocaleString();
    }

    resetSystem() {
        this.model = null;
        this.trainData = null;
        this.testData = null;
        
        document.getElementById('trainSamples').textContent = '0';
        document.getElementById('testSamples').textContent = '0';
        document.getElementById('layersCount').textContent = '0';
        document.getElementById('totalParams').textContent = '0';
        document.getElementById('logContainer').innerHTML = '';
        document.getElementById('randomTestResults').innerHTML = '';
        document.getElementById('singleTestResult').innerHTML = '';
        
        // 重置文件输入
        document.getElementById('trainFile').value = '';
        document.getElementById('testFile').value = '';
        document.getElementById('modelJsonFile').value = '';
        document.getElementById('modelWeightsFile').value = '';
        document.getElementById('singleSequence').value = '';
        
        document.getElementById('trainFileName').textContent = '未选择文件';
        document.getElementById('testFileName').textContent = '未选择文件';
        document.getElementById('modelJsonFileName').textContent = '未选择文件';
        document.getElementById('modelWeightsFileName').textContent = '未选择文件';

        this.log('系统已重置');
    }
}

// 初始化应用
let dnaClassifier;
document.addEventListener('DOMContentLoaded', () => {
    dnaClassifier = new DNAClassifier();
});
