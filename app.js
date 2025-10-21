class DNAClassifierApp {
    constructor() {
        this.model = null;
        this.isModelLoaded = false;
        this.chart = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.loadModel();
    }

    initializeElements() {
        this.dnaInput = document.getElementById('dna-sequence');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.clearBtn = document.getElementById('clear-btn');
        this.uploadBtn = document.getElementById('upload-btn');
        this.fileInput = document.getElementById('file-input');
        this.resultsSection = document.getElementById('results-section');
        this.loadingElement = document.getElementById('loading');
        this.seqLength = document.getElementById('seq-length');
        this.gcContent = document.getElementById('gc-content');
    }

    attachEventListeners() {
        this.analyzeBtn.addEventListener('click', () => this.analyzeSequence());
        this.clearBtn.addEventListener('click', () => this.clearInput());
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        this.dnaInput.addEventListener('input', () => this.updateSequenceInfo());
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                this.analyzeSequence();
            }
        });
    }

    async loadModel() {
        try {
            this.showLoading();
            console.log('Loading DNA classification model...');
            
            // Load the trained model
            this.model = await tf.loadLayersModel('models/model.json');
            this.isModelLoaded = true;
            
            console.log('Model loaded successfully');
            this.hideLoading();
        } catch (error) {
            console.error('Error loading model:', error);
            this.showError('Failed to load model. Please refresh the page.');
        }
    }

    updateSequenceInfo() {
        const sequence = this.dnaInput.value.trim().toUpperCase();
        const cleanSequence = sequence.replace(/[^ATCG]/g, '');
        
        this.seqLength.textContent = cleanSequence.length;
        
        if (cleanSequence.length > 0) {
            const gcCount = (cleanSequence.match(/[GC]/g) || []).length;
            const gcPercentage = ((gcCount / cleanSequence.length) * 100).toFixed(2);
            this.gcContent.textContent = `${gcPercentage}%`;
        } else {
            this.gcContent.textContent = '0%';
        }
    }

    async analyzeSequence() {
        const sequence = this.dnaInput.value.trim();
        
        if (!sequence) {
            this.showError('Please enter a DNA sequence');
            return;
        }

        if (!this.isModelLoaded) {
            this.showError('Model is still loading. Please wait...');
            return;
        }

        try {
            this.showLoading();
            
            // Validate and preprocess sequence
            const processedSequence = this.preprocessSequence(sequence);
            
            // Make prediction
            const predictions = await this.model.predict(processedSequence);
            const results = await predictions.data();
            
            // Display results
            this.displayResults(results, sequence);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Error analyzing sequence: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    preprocessSequence(sequence) {
        // Convert to uppercase and remove invalid characters
        const cleanSequence = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        
        if (cleanSequence.length < 50) {
            throw new Error('Sequence too short. Minimum length is 50 base pairs.');
        }
        
        if (cleanSequence.length > 10000) {
            throw new Error('Sequence too long. Maximum length is 10,000 base pairs.');
        }
        
        // One-hot encoding
        return this.oneHotEncode(cleanSequence);
    }

    oneHotEncode(sequence) {
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
        
        // Pad or truncate to fixed length (1000 bp)
        const fixedLength = 1000;
        while (encoded.length < fixedLength) {
            encoded.push([0, 0, 0, 0]); // Padding
        }
        
        if (encoded.length > fixedLength) {
            encoded.length = fixedLength; // Truncate
        }
        
        return tf.tensor3d([encoded]);
    }

    displayResults(predictions, originalSequence) {
        this.resultsSection.classList.remove('hidden');
        
        const classNames = ['Human', 'Bacteria', 'Virus', 'Plant'];
        const predictionData = Array.from(predictions);
        
        // Find top prediction
        const maxIndex = predictionData.indexOf(Math.max(...predictionData));
        const topPrediction = classNames[maxIndex];
        const topConfidence = (predictionData[maxIndex] * 100).toFixed(2);
        
        // Update top prediction display
        const topPredElement = document.getElementById('top-prediction');
        topPredElement.textContent = topPrediction;
        topPredElement.className = `prediction-label ${topPrediction.toLowerCase()}`;
        
        document.getElementById('confidence-score').textContent = `${topConfidence}%`;
        
        // Create confidence chart
        Visualization.createConfidenceChart(predictionData, classNames);
        
        // Show sequence analysis
        Visualization.displaySequenceAnalysis(originalSequence, topPrediction, topConfidence);
        
        // Show sequence features
        this.displaySequenceFeatures(originalSequence);
    }

    displaySequenceFeatures(sequence) {
        const cleanSequence = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        const featuresElement = document.getElementById('features-display');
        
        const gcCount = (cleanSequence.match(/[GC]/g) || []).length;
        const atCount = (cleanSequence.match(/[AT]/g) || []).length;
        const gcPercentage = ((gcCount / cleanSequence.length) * 100).toFixed(2);
        const atPercentage = ((atCount / cleanSequence.length) * 100).toFixed(2);
        
        const features = [
            { name: 'Total Length', value: `${cleanSequence.length} bp` },
            { name: 'GC Content', value: `${gcPercentage}%` },
            { name: 'AT Content', value: `${atPercentage}%` },
            { name: 'GC/AT Ratio', value: (gcCount / atCount).toFixed(2) },
            { name: 'A Count', value: (cleanSequence.match(/A/g) || []).length },
            { name: 'T Count', value: (cleanSequence.match(/T/g) || []).length },
            { name: 'C Count', value: (cleanSequence.match(/C/g) || []).length },
            { name: 'G Count', value: (cleanSequence.match(/G/g) || []).length }
        ];
        
        featuresElement.innerHTML = features.map(feature => 
            `<div class="feature-item">
                <span>${feature.name}:</span>
                <strong>${feature.value}</strong>
            </div>`
        ).join('');
    }

    showLoading() {
        this.loadingElement.classList.remove('hidden');
        this.analyzeBtn.disabled = true;
    }

    hideLoading() {
        this.loadingElement.classList.add('hidden');
        this.analyzeBtn.disabled = false;
    }

    showError(message) {
        alert(message); // In production, use a better error display
    }

    clearInput() {
        this.dnaInput.value = '';
        this.resultsSection.classList.add('hidden');
        this.updateSequenceInfo();
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const content = e.target.result;
                const sequence = this.parseSequenceFile(content, file.name);
                this.dnaInput.value = sequence;
                this.updateSequenceInfo();
            } catch (error) {
                this.showError('Error reading file: ' + error.message);
            }
        };
        
        reader.readAsText(file);
    }

    parseSequenceFile(content, filename) {
        if (filename.toLowerCase().endsWith('.fasta') || filename.toLowerCase().endsWith('.fa')) {
            // Parse FASTA format
            return content.split('\n')
                .filter(line => !line.startsWith('>'))
                .join('')
                .trim();
        } else if (filename.toLowerCase().endsWith('.csv')) {
            // Parse CSV - assuming first column is sequence
            const lines = content.split('\n');
            return lines[0].split(',')[0]; // Simple CSV parsing
        } else {
            // Plain text
            return content.trim();
        }
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DNAClassifierApp();
});
