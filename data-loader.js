class DataLoader {
    constructor() {
        this.samples = [];
        this.isDataLoaded = false;
        this.charToInt = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        this.riskLabels = ['High', 'Medium', 'Low'];
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
        
        // Return samples sorted by accuracy
        return this.samples.slice().sort((a, b) => {
            const aCorrect = a.isCorrect ? 1 : 0;
            const bCorrect = b.isCorrect ? 1 : 0;
            return bCorrect - aCorrect;
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
