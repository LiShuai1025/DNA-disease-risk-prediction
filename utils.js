class DNAUtils {
    static validateSequence(sequence) {
        if (!sequence || sequence.trim().length === 0) {
            throw new Error('Sequence cannot be empty');
        }

        const cleanSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        
        if (cleanSeq.length < 50) {
            throw new Error('Sequence must be at least 50 base pairs long');
        }

        if (cleanSeq.length > 10000) {
            throw new Error('Sequence cannot exceed 10,000 base pairs');
        }

        const validRatio = cleanSeq.length / sequence.length;
        if (validRatio < 0.8) {
            throw new Error('Sequence contains too many invalid characters. Only A, T, C, G are allowed.');
        }

        return cleanSeq;
    }

    static calculateGCContent(sequence) {
        const cleanSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        const gcCount = (cleanSeq.match(/[GC]/g) || []).length;
        return (gcCount / cleanSeq.length) * 100;
    }

    static calculateSequenceComplexity(sequence) {
        const cleanSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        const baseCounts = {
            'A': 0, 'T': 0, 'C': 0, 'G': 0
        };

        for (let base of cleanSeq) {
            baseCounts[base]++;
        }

        // Calculate Shannon entropy as a measure of complexity
        let entropy = 0;
        const total = cleanSeq.length;

        for (let base in baseCounts) {
            const probability = baseCounts[base] / total;
            if (probability > 0) {
                entropy -= probability * Math.log2(probability);
            }
        }

        return {
            entropy: entropy,
            maxEntropy: 2, // Maximum for 4 bases
            complexity: (entropy / 2) * 100, // Normalized percentage
            baseCounts: baseCounts
        };
    }

    static generateKmers(sequence, k = 3) {
        const kmers = {};
        const cleanSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');

        for (let i = 0; i <= cleanSeq.length - k; i++) {
            const kmer = cleanSeq.substring(i, i + k);
            kmers[kmer] = (kmers[kmer] || 0) + 1;
        }

        return kmers;
    }

    static formatSequence(sequence, lineLength = 80) {
        const cleanSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        let formatted = '';
        
        for (let i = 0; i < cleanSeq.length; i += lineLength) {
            formatted += cleanSeq.substring(i, i + lineLength) + '\n';
        }
        
        return formatted.trim();
    }

    static detectPatterns(sequence) {
        const patterns = {
            repeats: [],
            gcRich: [],
            atRich: [],
            palindromes: []
        };

        // Simple repeat detection
        const repeatRegex = /(A{4,}|T{4,}|C{4,}|G{4,})/gi;
        let match;
        while ((match = repeatRegex.exec(sequence)) !== null) {
            patterns.repeats.push({
                sequence: match[0],
                position: match.index,
                length: match[0].length
            });
        }

        // GC-rich regions (GC content > 60%)
        const gcRegex = /[GC]{5,}/gi;
        while ((match = gcRegex.exec(sequence)) !== null) {
            patterns.gcRich.push({
                sequence: match[0],
                position: match.index,
                length: match[0].length,
                gcContent: this.calculateGCContent(match[0])
            });
        }

        return patterns;
    }

    static async generateSequenceStats(sequence) {
        const cleanSeq = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        
        return {
            length: cleanSeq.length,
            gcContent: this.calculateGCContent(cleanSeq),
            complexity: this.calculateSequenceComplexity(cleanSeq),
            patterns: this.detectPatterns(cleanSeq),
            kmers: this.generateKmers(cleanSeq, 3)
        };
    }
}

// Utility functions for file handling
class FileUtils {
    static readFileAsText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(e);
            reader.readAsText(file);
        });
    }

    static downloadTextAsFile(text, filename) {
        const blob = new Blob([text], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Math utilities
class MathUtils {
    static softmax(arr) {
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
    }

    static normalize(arr) {
        const sum = arr.reduce((a, b) => a + b, 0);
        return arr.map(x => x / sum);
    }

    static standardize(arr) {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        const std = Math.sqrt(arr.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / arr.length);
        return arr.map(x => (x - mean) / std);
    }
}
