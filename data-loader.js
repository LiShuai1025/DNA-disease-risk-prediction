class DataLoader {
    constructor() {
        this.samples = [];
        this.isDataLoaded = false;
        this.charToInt = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        this.riskLabels = ['High', 'Medium', 'Low'];
    }

    // Load built-in dataset with realistic DNA sequences
    async loadBuiltInDataset(progressCallback = null) {
        return new Promise((resolve, reject) => {
            try {
                if (progressCallback) progressCallback(10);
                
                this.samples = this.generateRealisticDataset(300);
                
                if (progressCallback) progressCallback(100);
                
                this.isDataLoaded = true;
                console.log(`Built-in dataset loaded: ${this.samples.length} samples`);
                resolve(this.samples);
                
            } catch (error) {
                reject(error);
            }
        });
    }

    // Generate realistic DNA sequences with pathogenic markers
    generateRealisticDataset(count) {
        const samples = [];
        const pathogenicPatterns = [
            'CAGCAG', 'GCCGCC', 'CTGCAG', // Trinucleotide repeats
            'ATATAT', 'CGCGCG', // Dinucleotide repeats
            'GGG', 'CCC', 'AAA', 'TTT' // Homopolymer tracts
        ];
        
        for (let i = 1; i <= count; i++) {
            const isHighRisk = Math.random() < 0.3; // 30% high risk
            const isPathogenic = Math.random() < 0.25; // 25% pathogenic
            
            let sequence = this.generateDNASequence(100, isPathogenic, pathogenicPatterns);
            
            // Calculate features
            const features = this.calculateSequenceFeatures(sequence);
            
            // Determine risk based on features and patterns
            let actualRisk = 'Low';
            if (isHighRisk) {
                actualRisk = 'High';
            } else if (features.hasRepeats || features.gcContent > 55 || features.gcContent < 35) {
                actualRisk = 'Medium';
            }
            
            samples.push({
                id: `SAMPLE_${i}`,
                name: `DNA_${this.generateName()}`,
                sequence: sequence,
                features: features,
                actualRisk: actualRisk,
                predictedRisk: null,
                confidence: null,
                isCorrect: null,
                isPathogenic: isPathogenic
            });
        }
        
        return samples;
    }

    generateDNASequence(length, isPathogenic, pathogenicPatterns) {
        const bases = ['A', 'T', 'C', 'G'];
        let sequence = '';
        
        if (isPathogenic && Math.random() < 0.7) {
            // Insert pathogenic patterns
            const pattern = pathogenicPatterns[Math.floor(Math.random() * pathogenicPatterns.length)];
            const insertPos = Math.floor(Math.random() * (length - pattern.length));
            
            // Generate sequence with pathogenic pattern
            for (let i = 0; i < length; i++) {
                if (i >= insertPos && i < insertPos + pattern.length) {
                    sequence += pattern[i - insertPos];
                } else {
                    sequence += bases[Math.floor(Math.random() * 4)];
                }
            }
        } else {
            // Generate random sequence
            for (let i = 0; i < length; i++) {
                sequence += bases[Math.floor(Math.random() * 4)];
            }
        }
        
        return sequence;
    }

    calculateSequenceFeatures(sequence) {
        const gcCount = (sequence.match(/[GC]/g) || []).length;
        const gcContent = (gcCount / sequence.length) * 100;
        
        const numA = (sequence.match(/A/g) || []).length;
        const numT = (sequence.match(/T/g) || []).length;
        const numC = (sequence.match(/C/g) || []).length;
        const numG = (sequence.match(/G/g) || []).length;
        
        // Calculate k-mer complexity
        let kmerFreq = 0.5;
        if (sequence.length >= 3) {
            const kmers = new Set();
            for (let k = 0; k < sequence.length - 2; k++) {
                kmers.add(sequence.substring(k, k + 3));
            }
            kmerFreq = kmers.size / (sequence.length - 2);
        }
        
        // Detect repeats
        const hasRepeats = this.detectSignificantRepeats(sequence);
        
        return {
            gcContent: gcContent,
            sequenceLength: sequence.length,
            numA: numA,
            numT: numT,
            numC: numC,
            numG: numG,
            kmerFreq: kmerFreq,
            hasRepeats: hasRepeats
        };
    }

    detectSignificantRepeats(sequence) {
        let totalRepeats = 0;
        
        // Check for trinucleotide repeats
        for (let i = 0; i < sequence.length - 8; i++) {
            const triplet = sequence.substring(i, i + 3);
            let repeatCount = 1;
            
            for (let j = i + 3; j < sequence.length - 2; j += 3) {
                if (sequence.substring(j, j + 3) === triplet) {
                    repeatCount++;
                } else {
                    break;
                }
            }
            
            if (repeatCount >= 3) {
                totalRepeats++;
            }
        }
        
        return totalRepeats > 0;
    }

    // ... (rest of the methods remain the same as your original)
    // Keep all the existing file upload, CSV parsing, and other methods
}

const dataLoader = new DataLoader();
