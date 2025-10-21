class DNAClassifierApp {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        
        this.initializeElements();
        this.loadModel();
        this.setupEventListeners();
    }

    initializeElements() {
        this.dnaInput = document.getElementById('dna-input');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.fileInput = document.getElementById('file-input');
        this.resultsSection = document.getElementById('results-section');
        this.loadingElement = document.getElementById('loading');
    }

    async loadModel() {
        try {
            this.showLoading();
            this.model = await tf.loadLayersModel('models/dna_classifier.json');
            this.isModelLoaded = true;
            console.log('模型加载成功');
        } catch (error) {
            console.error('模型加载失败:', error);
            alert('模型加载失败，请刷新页面重试');
        } finally {
            this.hideLoading();
        }
    }

    setupEventListeners() {
        this.analyzeBtn.addEventListener('click', () => this.analyzeSequence());
        this.clearBtn.addEventListener('click', () => this.clearInput());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.analyzeSequence();
            }
        });
    }

    async analyzeSequence() {
        const sequence = this.dnaInput.value.trim();
        
        if (!sequence) {
            alert('请输入DNA序列');
            return;
        }

        if (!this.isModelLoaded) {
            alert('模型正在加载中，请稍后...');
            return;
        }

        try {
            this.showLoading();
            
            // 预处理序列
            const processedSeq = this.preprocessSequence(sequence);
            
            // 进行预测
            const prediction = await this.model.predict(processedSeq);
            const results = await prediction.data();
            
            // 可视化结果
            this.displayResults(results, sequence);
            
        } catch (error) {
            console.error('分析失败:', error);
            alert('分析过程中出现错误');
        } finally {
            this.hideLoading();
        }
    }

    preprocessSequence(sequence) {
        // 序列验证和预处理
        const validSequence = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        
        if (validSequence.length < 50) {
            throw new Error('序列长度过短，至少需要50个碱基');
        }

        // One-hot 编码
        return this.oneHotEncode(validSequence);
    }

    oneHotEncode(sequence) {
        // 实现DNA序列的one-hot编码
        const encoding = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1]
        };
        
        const encoded = [];
        for (let base of sequence) {
            encoded.push(encoding[base] || [0, 0, 0, 0]);
        }
        
        return tf.tensor3d([encoded]);
    }

    displayResults(prediction, originalSequence) {
        this.resultsSection.classList.remove('hidden');
        
        // 显示置信度图表
        Visualization.displayConfidenceChart(prediction);
        
        // 显示序列比对结果
        Visualization.displayAlignment(originalSequence, prediction);
    }

    showLoading() {
        this.loadingElement.classList.remove('hidden');
        this.analyzeBtn.disabled = true;
    }

    hideLoading() {
        this.loadingElement.classList.add('hidden');
        this.analyzeBtn.disabled = false;
    }

    clearInput() {
        this.dnaInput.value = '';
        this.resultsSection.classList.add('hidden');
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.dnaInput.value = this.parseFastaFile(e.target.result);
            };
            reader.readAsText(file);
        }
    }

    parseFastaFile(content) {
        // 简单的FASTA文件解析
        return content.split('\n')
            .filter(line => !line.startsWith('>'))
            .join('')
            .trim();
    }
}

// 初始化应用
document.addEventListener('DOMContentLoaded', () => {
    new DNAClassifierApp();
});
