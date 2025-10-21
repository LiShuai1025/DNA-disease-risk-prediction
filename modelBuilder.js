class ModelBuilder {
    static createModel(inputDim, outputDim, modelType = 'improved_dense') {
        switch (modelType) {
            case 'cnn':
                return this.createCNNModel(inputDim, outputDim);
            case 'improved_dense':
                return this.createImprovedDenseModel(inputDim, outputDim);
            case 'deep_dense':
                return this.createDeepDenseModel(inputDim, outputDim);
            default:
                return this.createImprovedDenseModel(inputDim, outputDim);
        }
    }

    static createImprovedDenseModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.dense({
            inputShape: [inputDim],
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));
        
        model.add(tf.layers.batchNormalization());
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        // Hidden layer
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
        
        // Output layer
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax',
            kernelInitializer: 'glorotNormal'
        }));

        // Compile model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    static createCNNModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        // Reshape for 1D convolution
        model.add(tf.layers.reshape({
            inputShape: [inputDim],
            targetShape: [inputDim, 1]
        }));
        
        // 1D convolutional layer
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
        
        // Fully connected layer
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        // Output layer
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
        
        // Deeper network architecture
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
        
        // Output layer
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax'
        }));

        model.compile({
            optimizer: tf.train.adam(0.0005), // Smaller learning rate
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }
}
