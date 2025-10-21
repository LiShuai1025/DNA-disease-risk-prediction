class DataLoader {
    static async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
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
            return [
                row.GC_Content || 0,
                row.AT_Content || 0,
                row.Sequence_Length || 0,
                row.Num_A || 0,
                row.Num_T || 0,
                row.Num_C || 0,
                row.Num_G || 0,
                row.kmer_3_freq || 0
            ];
        } catch (error) {
            console.error('特征提取错误:', error);
            return null;
        }
    }

    static extractFeaturesFromSequence(sequence) {
        const numA = (sequence.match(/A/g) || []).length;
        const numT = (sequence.match(/T/g) || []).length;
        const numC = (sequence.match(/C/g) || []).length;
        const numG = (sequence.match(/G/g) || []).length;
        const sequenceLength = sequence.length;

        const gcContent = (numG + numC) / sequenceLength;
        const atContent = (numA + numT) / sequenceLength;
        const kmer3Freq = this.calculateKmerFrequency(sequence, 3);

        return [gcContent, atContent, sequenceLength, numA, numT, numC, numG, kmer3Freq];
    }

    static calculateKmerFrequency(sequence, k) {
        const kmers = new Map();
        for (let i = 0; i <= sequence.length - k; i++) {
            const kmer = sequence.substring(i, i + k);
            kmers.set(kmer, (kmers.get(kmer) || 0) + 1);
        }
        return kmers.size > 0 ? sequence.length / kmers.size : 0;
    }

    static getClassLabelIndex(label) {
        const labels = ['Human', 'Bacteria', 'Virus', 'Plant'];
        return labels.indexOf(label);
    }
}
