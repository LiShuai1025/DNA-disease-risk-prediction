class ModelBuilder {
    static createModel(inputDim, outputDim, modelType = 'improved_dense') {
        switch (modelType) {
            case 'cnn':
                return this.createCNNModel(inputDim, outputDim);
            case 'improved_dense':
                return this.createImprovedDenseModel(inputDim, outputDim);
            case 'deep_dense':
                return this.createDeepDenseModel(inputDim, outputDim);
            case 'rnn':
                return this.createRNNModel(inputDim, outputDim);
            default:
                return this.createImprovedDenseModel(inputDim, outputDim);
        }
    }

    static createImprovedDenseModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            inputShape: [inputDim],
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax',
            kernelInitializer: 'glorotNormal'
        }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    static createCNNModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        model.add(tf.layers.reshape({
            inputShape: [inputDim],
            targetShape: [inputDim, 1]
        }));
        
        model.add(tf.layers.conv1d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.conv1d({
            filters: 32,
            kernelSize: 3,
            activation: 'relu',
            padding: 'same'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.globalMaxPooling1d());
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax'
        }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    static createDeepDenseModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            inputShape: [inputDim],
            units: 256,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.4 }));
        
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.4 }));
        
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax'
        }));

        model.compile({
            optimizer: tf.train.adam(0.0005),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    static createRNNModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        // Reshape input for RNN: [batch, timesteps, features]
        model.add(tf.layers.lstm({
            inputShape: [8, 1], // 8 timesteps, 1 feature per timestep
            units: 64,
            returnSequences: false,
            kernelInitializer: 'glorotNormal'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.2 }));
        
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax',
            kernelInitializer: 'glorotNormal'
        }));

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }
}
