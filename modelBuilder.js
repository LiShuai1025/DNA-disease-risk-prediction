class ModelBuilder {
    static createModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        // Input layer and hidden layers
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
        
        // Output layer
        model.add(tf.layers.dense({
            units: outputDim,
            activation: 'softmax'
        }));

        // Compile model
        model.compile({
            optimizer: 'adam',
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    static createCNNModel(inputDim, outputDim) {
        const model = tf.sequential();
        
        // Reshape input for CNN
        model.add(tf.layers.reshape({
            inputShape: [inputDim],
            targetShape: [inputDim, 1]
        }));
        
        // 1D Convolutional layers
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
        
        // Dense layers
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dropout({ rate: 0.3 }));
        
        // Output layer
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
