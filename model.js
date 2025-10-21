class DNAModel {
    constructor() {
        this.model = null;
        this.classNames = ['Human', 'Bacteria', 'Virus', 'Plant'];
    }

    async loadModel() {
        try {
            this.model = await tf.loadLayersModel('models/model.json');
            console.log('DNA classification model loaded successfully');
            return this.model;
        } catch (error) {
            console.error('Error loading model:', error);
            throw new Error('Failed to load the DNA classification model');
        }
    }

    async predict(sequenceTensor) {
        if (!this.model) {
            await this.loadModel();
        }
        
        const prediction = this.model.predict(sequenceTensor);
        const results = await prediction.data();
        
        // Clean up tensors to prevent memory leaks
        sequenceTensor.dispose();
        prediction.dispose();
        
        return results;
    }

    // Method to get model summary (for debugging)
    getModelSummary() {
        if (this.model) {
            this.model.summary();
        }
    }

    // Method to preprocess sequence for prediction
    preprocessSequence(sequence) {
        const encoding = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1]
        };
        
        const encoded = [];
        const cleanSequence = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        
        for (let base of cleanSequence) {
            encoded.push(encoding[base]);
        }
        
        // Pad/truncate to fixed length
        const fixedLength = 1000;
        while (encoded.length < fixedLength) {
            encoded.push([0, 0, 0, 0]);
        }
        
        if (encoded.length > fixedLength) {
            encoded.length = fixedLength;
        }
        
        return tf.tensor3d([encoded]);
    }
}
