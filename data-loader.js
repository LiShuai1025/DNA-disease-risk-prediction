class DataLoader {
    constructor() {
        this.samples = [];
        this.isDataLoaded = false;
        this.charToInt = {'A': 0, 'T': 1, 'C': 2, 'G': 3};
        this.riskLabels = ['High', 'Medium', 'Low'];
    }

    // 数据清洗和平衡
    cleanAndBalanceDataset(samples) {
        console.log('Cleaning and balancing dataset...');
        
        // 1. 清洗数据：移除无效样本
        const cleaned = samples.filter(sample => {
            if (!sample.sequence || sample.sequence.length < 10) return false;
            if (!sample.actualRisk || !this.riskLabels.includes(sample.actualRisk)) return false;
            return true;
        });
        
        console.log(`After cleaning: ${cleaned.length} samples`);
        
        // 2. 平衡数据集
        const riskCounts = {High: 0, Medium: 0, Low: 0};
        cleaned.forEach(s => riskCounts[s.actualRisk]++);
        
        const minCount = Math.min(...Object.values(riskCounts));
        const balanced = [];
        const currentCounts = {High: 0, Medium: 0, Low: 0};
        
        // 打乱顺序
        const shuffled = [...cleaned].sort(() => Math.random() - 0.5);
        
        for (const sample of shuffled) {
            if (currentCounts[sample.actualRisk] < minCount) {
                balanced.push(sample);
                currentCounts[sample.actualRisk]++;
            }
        }
        
        console.log(`Balanced dataset: ${balanced.length} samples`);
        console.log('Class distribution:', currentCounts);
        
        return balanced;
    }

    // 增强特征提取
    extractEnhancedFeatures(sequence) {
        const features = {};
        const seq = sequence.toUpperCase();
        
        // 基础特征
        const gcCount = (seq.match(/[GC]/g) || []).length;
        features.gcContent = (gcCount / seq.length) * 100;
        
        features.numA = (seq.match(/A/g) || []).length;
        features.numT = (seq.match(/T/g) || []).length;
        features.numC = (seq.match(/C/g) || []).length;
        features.numG = (seq.match(/G/g) || []).length;
        features.sequenceLength = seq.length;
        
        // 3-mer复杂度
        if (seq.length >= 3) {
            const kmers = new Set();
            for (let i = 0; i < seq.length - 2; i++) {
                kmers.add(seq.substring(i, i + 3));
            }
            features.kmerFreq = kmers.size / (seq.length - 2);
        } else {
            features.kmerFreq = 0.5;
        }
        
        // 序列熵
        features.entropy = this.calculateSequenceEntropy(seq);
        
        // 重复模式检测
        features.repeatScore = this.calculateRepeatScore(seq);
        
        // 二核苷酸频率特征
        const dinucleotideFeatures = this.calculateDinucleotideFeatures(seq);
        Object.assign(features, dinucleotideFeatures);
        
        return features;
    }

    calculateSequenceEntropy(sequence) {
        const freq = {};
        for (let base of sequence) {
            freq[base] = (freq[base] || 0) + 1;
        }
        
        let entropy = 0;
        const total = sequence.length;
        for (let base in freq) {
            const p = freq[base] / total;
            entropy -= p * Math.log2(p);
        }
        
        return entropy;
    }

    calculateRepeatScore(sequence) {
        let score = 0;
        
        // 检测2-4个碱基的重复
        for (let k = 2; k <= 4; k++) {
            for (let i = 0; i <= sequence.length - k * 3; i++) {
                const pattern = sequence.substring(i, i + k);
                let repeats = 1;
                
                for (let j = i + k; j <= sequence.length - k; j += k) {
                    if (sequence.substring(j, j + k) === pattern) {
                        repeats++;
                    } else {
                        break;
                    }
                }
                
                if (repeats >= 3) {
                    score += (repeats - 2); // 至少3次重复才计分
                }
            }
        }
        
        return Math.min(score, 10); // 限制最大分数
    }

    calculateDinucleotideFeatures(sequence) {
        const dinucleotides = ['AA', 'AT', 'AC', 'AG', 'TA', 'TT', 'TC', 'TG', 
                              'CA', 'CT', 'CC', 'CG', 'GA', 'GT', 'GC', 'GG'];
        const features = {};
        const total = Math.max(1, sequence.length - 1);
        
        // 计算每个二核苷酸的频率
        dinucleotides.forEach(dinuc => {
            const count = (sequence.match(new RegExp(dinuc, 'g')) || []).length;
            features[`dinuc_${dinuc}`] = count / total;
        });
        
        // 计算GC二核苷酸的总频率（重要特征）
        features.gcDinucTotal = features.dinuc_GC + features.dinuc_CG + features.dinuc_GG + features.dinuc_CC;
        
        return features;
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
                    
                    // 数据清洗和平衡
                    this.samples = this.cleanAndBalanceDataset(this.samples);
                    
                    // 增强特征提取
                    this.samples.forEach(sample => {
                        if (sample.sequence) {
                            const enhancedFeatures = this.extractEnhancedFeatures(sample.sequence);
                            sample.features = {...sample.features, ...enhancedFeatures};
                        }
                    });
                    
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

    // 其余方法保持不变...
    parseCSV(csvContent) {
        const results = Papa.parse(csvContent, {
            header: true,
            skipEmptyLines: true,
            transform: (value) => {
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

    processParsedData(parsedData) {
        this.samples = [];
        
        parsedData.forEach((row, index) => {
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
                sample.actualRisk = this.inferRiskFromClass(row.Class_Label);
            }
            
            // If no risk label is available, skip this sample
            if (!sample.actualRisk) {
                return;
            }
            
            // Calculate missing features from sequence if available
            if (sample.sequence && sample.sequence.length > 0) {
                this.calculateFeaturesFromSequence(sample);
            }
            
            this.samples.push(sample);
        });
        
        console.log(`Processed ${this.samples.length} samples`);
    }

    normalizeRiskLabel(label) {
        const labelStr = String(label).toLowerCase().trim();
        
        if (labelStr.includes('high') || labelStr === 'h') return 'High';
        if (labelStr.includes('medium') || labelStr.includes('moderate') || labelStr === 'm') return 'Medium';
        if (labelStr.includes('low') || labelStr === 'l') return 'Low';
        
        return null;
    }

    inferRiskFromClass(classLabel) {
        const classStr = String(classLabel).toLowerCase();
        
        if (classStr.includes('pathogen') || classStr.includes('cancer') || classStr.includes('disease')) {
            return 'High';
        } else if (classStr.includes('bacteria') || classStr.includes('virus')) {
            return 'Medium';
        } else {
            return 'Low';
        }
    }

    calculateFeaturesFromSequence(sample) {
        const sequence = sample.sequence.toUpperCase();
        
        if (sample.features.gcContent === undefined) {
            const gcCount = (sequence.match(/[GC]/g) || []).length;
            sample.features.gcContent = (gcCount / sequence.length) * 100;
        }
        
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
        
        if (sample.features.sequenceLength === undefined) {
            sample.features.sequenceLength = sequence.length;
        }
        
        if (sample.features.kmerFreq === undefined && sequence.length >= 3) {
            const kmers = new Set();
            for (let i = 0; i < sequence.length - 2; i++) {
                kmers.add(sequence.substring(i, i + 3));
            }
            sample.features.kmerFreq = kmers.size / (sequence.length - 2);
        }
    }

    generateName() {
        const prefixes = ['BRCA', 'TP53', 'EGFR', 'KRAS', 'BRAF', 'PTEN', 'APC', 'VHL'];
        const suffixes = ['001', '002', '003', '004', '005', '006', '007', '008'];
        return prefixes[Math.floor(Math.random() * prefixes.length)] + 
               suffixes[Math.floor(Math.random() * suffixes.length)];
    }

    getRankedSamples() {
        if (!this.samples || this.samples.length === 0) {
            return [];
        }
        
        if (!this.samples.some(s => s.predictedRisk !== null)) {
            return this.samples.slice().sort((a, b) => a.id.localeCompare(b.id));
        }
        
        return this.samples.slice().sort((a, b) => {
            const aCorrect = a.isCorrect ? 1 : 0;
            const bCorrect = b.isCorrect ? 1 : 0;
            
            if (aCorrect !== bCorrect) {
                return bCorrect - aCorrect;
            }
            
            if (a.confidence && b.confidence) {
                return b.confidence - a.confidence;
            }
            
            return 0;
        });
    }

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
