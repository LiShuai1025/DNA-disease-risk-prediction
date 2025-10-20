class DataLoader {
    constructor() {
        this.samples = [];
        this.trainingData = null;
        this.testingData = null;
        this.charToInt = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        this.riskLabels = ['High', 'Medium', 'Low'];
        this.isDataLoaded = false;
    }

    // Load sample DNA data
    async loadSampleData() {
        // In a real application, this would load actual data from a server
        // For now, using generated data
        this.samples = this.generateSampleData(30); // Reduced sample count for performance
        this.isDataLoaded = true;
        return this.samples;
    }

    // Generate sample data
    generateSampleData(count) {
        const samples = [];
        const bases = ['A', 'T', 'C', 'G'];
        
        for (let i = 1; i <= count; i++) {
            // Generate random DNA sequence
            let sequence = '';
            for (let j = 0; j < 100; j++) {
                sequence += bases[Math.floor(Math.random() * 4)];
            }
            
            // Generate features
            const gcContent = 30 + Math.random() * 40; // 30-70%
            const numA = Math.floor(Math.random() * 30) + 15;
            const numT = Math.floor(Math.random() * 30) + 15;
            const numC = Math.floor(Math.random() * 30) + 15;
            const numG = 100 - numA - numT - numC;
            const kmerFreq = Math.random();
            
            // Generate risk level based on features (simulating real patterns)
            let riskLevel;
            const riskScore = (gcContent / 100) + (kmerFreq * 0.3) + (Math.random() * 0.3);
            
            if (riskScore > 0.7) riskLevel = 'High';
            else if (riskScore > 0.4) riskLevel = 'Medium';
            else riskLevel = 'Low';
            
            samples.push({
                id: `SAMPLE_${i}`,
                name: `DNA_${this.generateName()}`,
                sequence: sequence,
                features: {
                    gcContent: gcContent,
                    sequenceLength: 100,
                    numA: numA,
                    numT: numT,
                    numC: numC,
                    numG: numG,
                    kmerFreq: kmerFreq
                },
                actualRisk: riskLevel,
                predictedRisk: null,
                confidence: null,
                isCorrect: null
            });
        }
        
        return samples;
    }

    // Generate random DNA sample names
    generateName() {
        const prefixes = ['BRCA', 'TP53', 'EGFR', 'KRAS', 'BRAF', 'PTEN', 'APC', 'VHL'];
        const suffixes = ['001', '002', '003', '004', '005', '006', '007', '008'];
        return prefixes[Math.floor(Math.random() * prefixes.length)] + 
               suffixes[Math.floor(Math.random() * suffixes.length)];
    }

    // Encode DNA sequence
    encodeDNASequence(sequence) {
        const cleanedSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        return Array.from(cleanedSeq).map(char => this.charToInt[char]);
    }

    // Normalize features
    normalizeFeatures(features) {
        const normalized = {};
        
        normalized.gcContent = (features.gcContent - 30) / 40;
        normalized.sequenceLength = (features.sequenceLength - 50) / 100;
        
        const totalBases = features.numA + features.numT + features.numC + features.numG;
        normalized.numA = features.numA / totalBases;
        normalized.numT = features.numT / totalBases;
        normalized.numC = features.numC / totalBases;
        normalized.numG = features.numG / totalBases;
        
        normalized.kmerFreq = features.kmerFreq;
        
        return Object.values(normalized);
    }

    // Prepare training data
    prepareTrainingData() {
        const sequences = [];
        const numericalFeatures = [];
        const labels = [];
        
        this.samples.forEach(sample => {
            sequences.push(this.encodeDNASequence(sample.sequence));
            numericalFeatures.push(this.normalizeFeatures(sample.features));
            labels.push(this.riskLabels.indexOf(sample.actualRisk));
        });
        
        this.trainingData = {
            sequences: sequences,
            numericalFeatures: numericalFeatures,
            labels: labels
        };
        
        return this.trainingData;
    }

    // Split data into training and testing sets
    splitData(trainRatio = 0.7) {
        if (!this.trainingData) this.prepareTrainingData();
        
        const totalSamples = this.samples.length;
        const trainSize = Math.floor(totalSamples * trainRatio);
        
        // Shuffle indices randomly
        const indices = Array.from({length: totalSamples}, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        
        const trainIndices = indices.slice(0, trainSize);
        const testIndices = indices.slice(trainSize);
        
        this.testingData = {
            sequences: testIndices.map(i => this.trainingData.sequences[i]),
            numericalFeatures: testIndices.map(i => this.trainingData.numericalFeatures[i]),
            labels: testIndices.map(i => this.trainingData.labels[i]),
            sampleIndices: testIndices
        };
        
        return this.testingData;
    }

    // Get samples ranked by accuracy
    getRankedSamples() {
        if (!this.samples.some(s => s.predictedRisk !== null)) {
            // If no prediction data, return sorted by ID
            return this.samples.slice().sort((a, b) => a.id.localeCompare(b.id));
        }
        
        // Return samples sorted by accuracy
        return this.samples.slice().sort((a, b) => {
            const aCorrect = a.isCorrect ? 1 : 0;
            const bCorrect = b.isCorrect ? 1 : 0;
            return bCorrect - aCorrect;
        });
    }

    // Get timeline data for chart
    getTimelineData() {
        const predictions = this.samples
            .filter(s => s.predictedRisk !== null)
            .map((sample, index) => ({
                x: index,
                y: sample.isCorrect ? 1 : 0,
                sample: sample
            }));
        
        return predictions;
    }
}

const dataLoader = new DataLoader();
