class DataLoader {
    constructor() {
        this.samples = [];
        this.isDataLoaded = false;
        this.charToInt = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        this.riskLabels = ['High', 'Medium', 'Low'];
    }

    // Load built-in dataset for training
    async loadBuiltInDataset(progressCallback = null) {
        return new Promise((resolve, reject) => {
            try {
                if (progressCallback) progressCallback(10);
                
                // Generate realistic synthetic DNA data based on your original dataset
                this.samples = this.generateRealisticDataset(200); // Larger dataset for better training
                
                if (progressCallback) progressCallback(100);
                
                this.isDataLoaded = true;
                console.log(`Built-in dataset loaded: ${this.samples.length} samples`);
                resolve(this.samples);
                
            } catch (error) {
                reject(error);
            }
        });
    }

    // Generate realistic DNA dataset based on patterns from your original data
    generateRealisticDataset(count) {
        const samples = [];
        const bases = ['A', 'T', 'C', 'G'];
        
        // Define patterns for different risk categories based on your original data analysis
        const riskPatterns = {
            'High': {
                gcRange: [48, 62],    // Higher GC content
                kmerRange: [0.3, 0.7], // Moderate complexity
                baseBias: 0.3         // Some base composition bias
            },
            'Medium': {
                gcRange: [42, 58],    // Medium GC content  
                kmerRange: [0.4, 0.8], // Higher complexity
                baseBias: 0.2         // Less bias
            },
            'Low': {
                gcRange: [38, 52],    // Lower GC content
                kmerRange: [0.2, 0.6], // Lower complexity
                baseBias: 0.4         // More bias
            }
        };
        
        for (let i = 1; i <= count; i++) {
            // Assign risk category with balanced distribution
            const riskLevel = this.riskLabels[i % 3];
            const pattern = riskPatterns[riskLevel];
            
            // Generate DNA sequence with characteristics matching the risk pattern
            let sequence = '';
            const gcTarget = pattern.gcRange[0] + Math.random() * (pattern.gcRange[1] - pattern.gcRange[0]);
            
            // Generate sequence with controlled GC content
            let gcCount = 0;
            const sequenceLength = 100;
            
            for (let j = 0; j < sequenceLength; j++) {
                let base;
                const currentGCPercent = (gcCount / (j + 1)) * 100;
                
                if (currentGCPercent < gcTarget) {
                    // Need more GC
                    base = Math.random() < 0.7 ? (Math.random() < 0.5 ? 'G' : 'C') : bases[Math.floor(Math.random() * 4)];
                } else {
                    // Need more AT
                    base = Math.random() < 0.7 ? (Math.random() < 0.5 ? 'A' : 'T') : bases[Math.floor(Math.random() * 4)];
                }
                
                if (base === 'G' || base === 'C') gcCount++;
                sequence += base;
            }
            
            // Calculate features
            const gcContent = (gcCount / sequenceLength) * 100;
            const numA = (sequence.match(/A/g) || []).length;
            const numT = (sequence.match(/T/g) || []).length;
            const numC = (sequence.match(/C/g) || []).length;
            const numG = (sequence.match(/G/g) || []).length;
            
            // Calculate k-mer frequency with pattern-based adjustment
            let kmerFreq;
            if (sequence.length >= 3) {
                const kmers = new Set();
                for (let k = 0; k < sequence.length - 2; k++) {
                    kmers.add(sequence.substring(k, k + 3));
                }
                const baseKmerFreq = kmers.size / (sequence.length - 2);
                // Adjust based on risk pattern
                kmerFreq = pattern.kmerRange[0] + (baseKmerFreq * (pattern.kmerRange[1] - pattern.kmerRange[0]));
            } else {
                kmerFreq = 0.5;
            }
            
            samples.push({
                id: `SAMPLE_${i}`,
                name: `DNA_${this.generateName()}`,
                sequence: sequence,
                features: {
                    gcContent: gcContent,
                    sequenceLength: sequenceLength,
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
        
        console.log(`Generated realistic dataset with ${samples.length} samples`);
        return samples;
    }

    // Load data from uploaded file
    async loadFromFile(file, progressCallback = null) {
        return new Promise((resolve, reject) => {
            if (progressCallback) progressCallback(10);
            
            const reader = new FileReader();
            
            reader.onload = (e) => {
                try {
                    if (progressCallback) progressCallback(30);
                    
                    const fileContent = e.target.result;
                    let parsedData;
                    
                    if (file.name.endsWith('.csv')) {
                        parsedData = this.parseCSV(fileContent);
                    } else {
                        throw new Error('Unsupported file format. Please upload a CSV file.');
                    }
                    
                    if (progressCallback) progressCallback(70);
                    
                    // Process the parsed data
                    this.processParsedData(parsedData);
                    
                    if (progressCallback) progressCallback(100);
                    this.isDataLoaded = true;
                    resolve(this.samples);
                    
                } catch (error) {
                    reject(error);
                }
            };
            
            reader.onerror = () => {
                reject(new Error('Failed to read file'));
            };
            
            if (file.name.endsWith('.csv')) {
                reader.readAsText(file);
            } else {
                reject(new Error('Unsupported file format'));
            }
        });
    }

    // Parse CSV content
    parseCSV(csvContent) {
        const results = Papa.parse(csvContent, {
            header: true,
            skipEmptyLines: true,
            transform: (value) => {
                // Convert numeric strings to numbers
                if (!isNaN(value) && value.trim() !== '') {
                    return parseFloat(value);
                }
                return value;
            }
        });
        
        if (results.errors.length > 0) {
            throw new Error(`CSV parsing error: ${results.errors[0].message}`);
        }
        
        if (results.data.length === 0) {
            throw new Error('No data found in CSV file');
        }
        
        return results.data;
    }

    // Process parsed data into samples
    processParsedData(parsedData) {
        this.samples = [];
        
        parsedData.forEach((row, index) => {
            // Determine available columns and map them
            const sample = {
                id: row.Sample_ID || `SAMPLE_${index + 1}`,
                name: row.Sample_ID || `DNA_${this.generateName()}`,
                sequence: row.Sequence || '',
                features: {},
                actualRisk: null,
                predictedRisk: null,
                confidence: null,
                isCorrect: null
            };
            
            // Extract features from available columns
            if (row.GC_Content !== undefined) sample.features.gcContent = parseFloat(row.GC_Content);
            if (row.Sequence_Length !== undefined) sample.features.sequenceLength = parseInt(row.Sequence_Length);
            if (row.Num_A !== undefined) sample.features.numA = parseInt(row.Num_A);
            if (row.Num_T !== undefined) sample.features.numT = parseInt(row.Num_T);
            if (row.Num_C !== undefined) sample.features.numC = parseInt(row.Num_C);
            if (row.Num_G !== undefined) sample.features.numG = parseInt(row.Num_G);
            if (row.kmer_3_freq !== undefined) sample.features.kmerFreq = parseFloat(row.kmer_3_freq);
            
            // Extract risk label
            if (row.Disease_Risk) {
                sample.actualRisk = this.normalizeRiskLabel(row.Disease_Risk);
            } else if (row.Class_Label) {
                // Try to infer risk from class label
                sample.actualRisk = this.inferRiskFromClass(row.Class_Label);
            }
            
            // If no risk label is available, generate a random one for demo purposes
            if (!sample.actualRisk) {
                sample.actualRisk = this.riskLabels[Math.floor(Math.random() * this.riskLabels.length)];
            }
            
            // Calculate missing features from sequence if available
            if (sample.sequence && sample.sequence.length > 0) {
                this.calculateFeaturesFromSequence(sample);
            }
            
            this.samples.push(sample);
        });
        
        console.log(`Processed ${this.samples.length} samples`);
    }

    // Normalize risk labels
    normalizeRiskLabel(label) {
        const labelStr = String(label).toLowerCase().trim();
        
        if (labelStr.includes('high') || labelStr === 'h') return 'High';
        if (labelStr.includes('medium') || labelStr.includes('moderate') || labelStr === 'm') return 'Medium';
        if (labelStr.includes('low') || labelStr === 'l') return 'Low';
        
        return null;
    }

    // Infer risk from class label
    inferRiskFromClass(classLabel) {
        const classStr = String(classLabel).toLowerCase();
        
        // Simple heuristic - in real application, this would be based on domain knowledge
        if (classStr.includes('pathogen') || classStr.includes('cancer') || classStr.includes('disease')) {
            return 'High';
        } else if (classStr.includes('bacteria') || classStr.includes('virus')) {
            return 'Medium';
        } else {
            return 'Low';
        }
    }

    // Calculate features from DNA sequence
    calculateFeaturesFromSequence(sample) {
        const sequence = sample.sequence.toUpperCase();
        
        // Calculate GC content if not provided
        if (sample.features.gcContent === undefined) {
            const gcCount = (sequence.match(/[GC]/g) || []).length;
            sample.features.gcContent = (gcCount / sequence.length) * 100;
        }
        
        // Calculate base counts if not provided
        if (sample.features.numA === undefined) {
            sample.features.numA = (sequence.match(/A/g) || []).length;
        }
        if (sample.features.numT === undefined) {
            sample.features.numT = (sequence.match(/T/g) || []).length;
        }
        if (sample.features.numC === undefined) {
            sample.features.numC = (sequence.match(/C/g) || []).length;
        }
        if (sample.features.numG === undefined) {
            sample.features.numG = (sequence.match(/G/g) || []).length;
        }
        
        // Set sequence length if not provided
        if (sample.features.sequenceLength === undefined) {
            sample.features.sequenceLength = sequence.length;
        }
        
        // Calculate k-mer frequency if not provided
        if (sample.features.kmerFreq === undefined && sequence.length >= 3) {
            const kmers = new Set();
            for (let i = 0; i < sequence.length - 2; i++) {
                kmers.add(sequence.substring(i, i + 3));
            }
            sample.features.kmerFreq = kmers.size / (sequence.length - 2);
        }
    }

    // Generate random DNA sample names
    generateName() {
        const prefixes = ['BRCA', 'TP53', 'EGFR', 'KRAS', 'BRAF', 'PTEN', 'APC', 'VHL'];
        const suffixes = ['001', '002', '003', '004', '005', '006', '007', '008'];
        return prefixes[Math.floor(Math.random() * prefixes.length)] + 
               suffixes[Math.floor(Math.random() * suffixes.length)];
    }

    // Get samples ranked by accuracy
    getRankedSamples() {
        if (!this.samples || this.samples.length === 0) {
            return [];
        }
        
        if (!this.samples.some(s => s.predictedRisk !== null)) {
            // If no prediction data, return sorted by ID
            return this.samples.slice().sort((a, b) => a.id.localeCompare(b.id));
        }
        
        // Return samples sorted by accuracy (correct predictions first)
        return this.samples.slice().sort((a, b) => {
            const aCorrect = a.isCorrect ? 1 : 0;
            const bCorrect = b.isCorrect ? 1 : 0;
            
            // First sort by correctness
            if (aCorrect !== bCorrect) {
                return bCorrect - aCorrect;
            }
            
            // Then sort by confidence (if available)
            if (a.confidence && b.confidence) {
                return b.confidence - a.confidence;
            }
            
            return 0;
        });
    }

    // Get timeline data for chart
    getTimelineData() {
        if (!this.samples || this.samples.length === 0) {
            return [];
        }
        
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
