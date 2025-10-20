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

    // Load data from uploaded file - ADDING THE MISSING METHOD
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

    // Generate realistic DNA dataset with pathogenic markers
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
