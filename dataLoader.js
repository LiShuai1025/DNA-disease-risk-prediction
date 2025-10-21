class DataLoader {
    static async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                delimiter: '\t', // 改为制表符分隔
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        reject(new Error(results.errors[0].message));
                    } else {
                        resolve(results.data);
                    }
                },
                error: (error) => {
                    reject(error);
                }
            });
        });
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
            // 处理可能的数值格式问题
            return [
                parseFloat(row.GC_Content) || 0,
                parseFloat(row.AT_Content) || 0,
                parseInt(row.Sequence_Length) || 0,
                parseInt(row.Num_A) || 0,
                parseInt(row.Num_T) || 0,
                parseInt(row.Num_C) || 0,
                parseInt(row.Num_G) || 0,
                parseFloat(row.kmer_3_freq) || 0
            ];
        } catch (error) {
            console.error('Feature extraction error:', error, row);
            return null;
        }
    }

    static extractFeaturesFromSequence(sequence) {
        // 清理序列（移除数字和其他非ATCG字符）
        const cleanSequence = sequence.replace(/[^ATCG]/gi, '').toUpperCase();
        
        const numA = (cleanSequence.match(/A/g) || []).length;
        const numT = (cleanSequence.match(/T/g) || []).length;
        const numC = (cleanSequence.match(/C/g) || []).length;
        const numG = (cleanSequence.match(/G/g) || []).length;
        const sequenceLength = cleanSequence.length;

        const gcContent = sequenceLength > 0 ? (numG + numC) / sequenceLength : 0;
        const atContent = sequenceLength > 0 ? (numA + numT) / sequenceLength : 0;
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
        return kmers.size > 0 ? sequence.length / kmers.size : 0;
    }

    static getClassLabelIndex(label) {
        const labels = ['Human', 'Bacteria', 'Virus', 'Plant'];
        const cleanLabel = label.trim();
        return labels.indexOf(cleanLabel);
    }
}
