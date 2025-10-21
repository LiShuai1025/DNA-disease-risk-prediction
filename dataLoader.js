class DataLoader {
    static async loadCSV(file, delimiter = '\t') {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                delimiter: delimiter,
                skipEmptyLines: true,
                transform: function(value) {
                    // 清理数据值
                    return typeof value === 'string' ? value.trim() : value;
                },
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error(results.errors[0].message));
                    } else {
                        // 验证数据格式
                        const validation = this.validateDataFormat(results.data);
                        if (!validation.isValid) {
                            reject(new Error(validation.message));
                        } else {
                            resolve(results.data);
                        }
                    }
                },
                error: (error) => {
                    reject(error);
                }
            });
        });
    }

    static validateDataFormat(data) {
        if (!data || data.length === 0) {
            return { isValid: false, message: 'No data found in file' };
        }

        const firstRow = data[0];
        const requiredFields = ['Sequence', 'Class_Label'];
        const missingFields = requiredFields.filter(field => !(field in firstRow));

        if (missingFields.length > 0) {
            return { 
                isValid: false, 
                message: `Missing required fields: ${missingFields.join(', ')}. Found fields: ${Object.keys(firstRow).join(', ')}` 
            };
        }

        let validCount = 0;
        for (const row of data) {
            if (row.Sequence && row.Class_Label) {
                validCount++;
            }
        }

        return {
            isValid: validCount > 0,
            message: `Found ${validCount} valid samples out of ${data.length} total rows`,
            validCount: validCount
        };
    }

    static processData(rawData) {
        const features = [];
        const labels = [];
        const processedData = [];

        for (const row of rawData) {
            if (!row.Sequence || !row.Class_Label) continue;

            const featureVector = this.extractFeaturesFromRow(row);
            const labelIndex = this.getClassLabelIndex(row.Class_Label);

            if (featureVector && labelIndex !== -1) {
                features.push(featureVector);
                labels.push(labelIndex);
                processedData.push(row);
            }
        }

        return { features, labels, rawData: processedData };
    }

    static extractFeaturesFromRow(row) {
        try {
            return [
                this.parseNumber(row.GC_Content),
                this.parseNumber(row.AT_Content),
                this.parseNumber(row.Sequence_Length),
                this.parseNumber(row.Num_A),
                this.parseNumber(row.Num_T),
                this.parseNumber(row.Num_C),
                this.parseNumber(row.Num_G),
                this.parseNumber(row.kmer_3_freq)
            ];
        } catch (error) {
            console.error('Feature extraction error:', error, row);
            return null;
        }
    }

    static parseNumber(value) {
        if (typeof value === 'number') return value;
        if (typeof value === 'string') {
            const parsed = parseFloat(value);
            return isNaN(parsed) ? 0 : parsed;
        }
        return 0;
    }

    static extractFeaturesFromSequence(sequence) {
        // 清理序列（移除数字和其他非ATCG字符）
        const cleanSequence = sequence.replace(/[^ATCGatcg]/g, '').toUpperCase();
        
        if (cleanSequence.length === 0) {
            return [0, 0, 0, 0, 0, 0, 0, 0];
        }
        
        const numA = (cleanSequence.match(/A/g) || []).length;
        const numT = (cleanSequence.match(/T/g) || []).length;
        const numC = (cleanSequence.match(/C/g) || []).length;
        const numG = (cleanSequence.match(/G/g) || []).length;
        const sequenceLength = cleanSequence.length;

        const gcContent = (numG + numC) / sequenceLength;
        const atContent = (numA + numT) / sequenceLength;
        const kmer3Freq = this.calculateKmerFrequency(cleanSequence, 3);

        return [gcContent, atContent, sequenceLength, numA, numT, numC, numG, kmer3Freq];
    }

    static calculateKmerFrequency(sequence, k) {
        if (sequence.length < k) return 0;
        
        const kmers = new Map();
        for (let i = 0; i <= sequence.length - k; i++) {
            const kmer = sequence.substring(i, i + k);
            kmers.set(kmer, (kmers.get(kmer) || 0) + 1);
        }
        return kmers.size > 0 ? (sequence.length - k + 1) / kmers.size : 0;
    }

    static getClassLabelIndex(label) {
        const labels = ['Human', 'Bacteria', 'Virus', 'Plant'];
        const cleanLabel = String(label).trim();
        return labels.indexOf(cleanLabel);
    }
}
