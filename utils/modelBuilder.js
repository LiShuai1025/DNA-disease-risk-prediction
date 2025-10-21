class ModelBuilder {
    static createModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        // 输入层和隐藏层
        model.add(tf.layers.dense({
            inputShape: [inputDim],
            units: 64,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        model.add(tf.layers.dense({
            units: 16,
            activation: 'relu'
        }));
        
        // 输出层
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax'
        }));

        // 编译模型
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    static createCNNModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        // 重塑输入为2D格式 (用于CNN)
        model.add(tf.layers.reshape({
            inputShape: [inputDim],
            targetShape: [inputDim, 1]
        }));
        
        // 1D卷积层
        model.add(tf.layers.conv1d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu'
        }));
        
        model.add(tf.layers.maxPooling1d({
            poolSize: 2
        }));
        
        model.add(tf.layers.conv1d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu'
        }));
        
        model.add(tf.layers.globalMaxPooling1d());
        
        // 全连接层
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        // 输出层
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax'
        }));

        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }
}
