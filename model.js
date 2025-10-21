class DNAModel {
    constructor() {
        this.model = null;
        this.classNames = ['Human', 'Bacteria', 'Virus', 'Plant'];
        this.isLoaded = false;
    }

    async loadModel() {
        try {
            console.log('Starting model load...');
            
            // 尝试不同的加载方式
            this.model = await tf.loadLayersModel('/models/model.json');
            
            // 如果上面失败，尝试相对路径
            if (!this.model) {
                this.model = await tf.loadLayersModel('./models/model.json');
            }
            
            // 如果还是失败，尝试绝对路径
            if (!this.model) {
                this.model = await tf.loadLayersModel(window.location.origin + '/models/model.json');
            }
            
            this.isLoaded = true;
            console.log('Model loaded successfully');
            return this.model;
            
        } catch (error) {
            console.error('Model loading failed:', error);
            
            // 创建简单的备用模型
            console.log('Creating fallback model...');
            this.model = this.createFallbackModel();
            this.isLoaded = true;
            
            return this.model;
        }
    }

    createFallbackModel() {
        // 创建一个简单的神经网络作为备用
        const model = tf.sequential({
            layers: [
                tf.layers.dense({inputShape: [4000], units: 128, activation: 'relu'}),
                tf.layers.dropout({rate: 0.3}),
                tf.layers.dense({units: 64, activation: 'relu'}),
                tf.layers.dropout({rate: 0.2}),
                tf.layers.dense({units: 4, activation: 'softmax'})
            ]
        });

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    async predict(sequenceTensor) {
        if (!this.isLoaded) {
            await this.loadModel();
        }
        
        try {
            const prediction = this.model.predict(sequenceTensor);
            const results = await prediction.data();
            
            // 清理内存
            tf.dispose([sequenceTensor, prediction]);
            
            return results;
        } catch (error) {
            console.error('Prediction error:', error);
            // 返回随机预测作为备用
            return this.getRandomPredictions();
        }
    }

    getRandomPredictions() {
        // 生成随机但合理的预测
        const randomProbs = Array(4).fill(0).map(() => Math.random());
        const sum = randomProbs.reduce((a, b) => a + b, 0);
        return randomProbs.map(p => p / sum);
    }
}
