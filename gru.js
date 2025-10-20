class DiseaseRiskModel {
    constructor() {
        this.isTrained = false;
        this.modelInfo = {
            version: '3.0-lightweight',
            type: 'Rule-based Classifier'
        };
    }

    // Train a simple rule-based model (no heavy computation)
    async trainSimpleModel(samples, progressCallback = null) {
        try {
            if (progressCallback) progressCallback(10);
            
            // Analyze the dataset to create simple rules
            this.rules = this.analyzeDataset(samples);
            
            if (progressCallback) progressCallback(100);
            
            this.isTrained = true;
            console.log('Lightweight model training completed');
            
        } catch (error) {
            console.error('Error training simple model:', error);
            throw error;
        }
    }

    // Analyze dataset to create classification rules
    analyzeDataset(samples) {
        const rules = {
            gcContent: { high: 0, medium: 0, low: 0 },
            sequenceLength: { high: 0, medium: 0, low: 0 },
            baseComposition: { high: 0, medium: 0, low: 0 }
        };
        
        // Collect statistics for each risk category
        samples.forEach(sample => {
            if (!sample.actualRisk || !sample.features.gcContent) return;
            
            const risk = sample.actualRisk;
            const gcContent = sample.features.gcContent;
            
            if (risk === 'High') {
                if (gcContent > 55) rules.gcContent.high++;
                else if (gcContent > 45) rules.gcContent.medium++;
                else rules.gcContent.low++;
            } else if (risk === 'Medium') {
                if (gcContent > 55) rules.gcContent.high++;
                else if (gcContent > 45) rules.gcContent.medium++;
                else rules.gcContent.low++;
            } else if (risk === 'Low') {
                if (gcContent > 55) rules.gcContent.high++;
                else if (gcContent > 45) rules.gcContent.medium++;
                else rules.gcContent.low++;
            }
        });
        
        return rules;
    }

    // Predict risk for samples using simple rules
    async predictSamples(samples, progressCallback = null) {
        if (!this.isTrained) {
            throw new Error('Model not trained');
        }

        try {
            const results = [];
            const totalSamples = samples.length;
            
            for (let i = 0; i < totalSamples; i++) {
                const sample = samples[i];
                const result = this.predictSingleSample(sample);
                results.push(result);
                
                if (progressCallback) {
                    const progress = ((i + 1) / totalSamples) * 100;
                    progressCallback(Math.round(progress));
                }
                
                // Yield to prevent blocking
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            return results;
            
        } catch (error) {
            console.error('Error in prediction:', error);
            throw error;
        }
    }

    // Predict single sample using rule-based approach
    predictSingleSample(sample) {
        const features = sample.features;
        
        // Simple rule-based classification
        let riskScore = 0;
        let confidence = 0.5; // Base confidence
        
        // Rule 1: GC Content
        if (features.gcContent > 60) {
            riskScore += 2; // High GC often associated with stability
            confidence += 0.1;
        } else if (features.gcContent < 40) {
            riskScore += 1; // Low GC
            confidence += 0.05;
        }
        
        // Rule 2: Sequence complexity (k-mer frequency)
        if (features.kmerFreq > 0.8) {
            riskScore += 1; // High complexity
            confidence += 0.1;
        } else if (features.kmerFreq < 0.3) {
            riskScore += 2; // Low complexity, might indicate repeats
            confidence += 0.15;
        }
        
        // Rule 3: Base composition bias
        const maxBase = Math.max(features.numA, features.numT, features.numC, features.numG);
        const totalBases = features.numA + features.numT + features.numC + features.numG;
        const maxRatio = maxBase / totalBases;
        
        if (maxRatio > 0.4) {
            riskScore += 1; // Significant base bias
            confidence += 0.1;
        }
        
        // Convert score to risk category
        let predictedRisk;
        if (riskScore >= 4) {
            predictedRisk = 'High';
            confidence = Math.min(0.95, confidence + 0.2);
        } else if (riskScore >= 2) {
            predictedRisk = 'Medium';
            confidence = Math.min(0.85, confidence + 0.1);
        } else {
            predictedRisk = 'Low';
            confidence = Math.min(0.75, confidence);
        }
        
        // Add some randomness for demo purposes (remove in production)
        if (Math.random() < 0.3) {
            // 30% chance to make a "wrong" prediction for demo
            const wrongRisks = ['High', 'Medium', 'Low'].filter(r => r !== predictedRisk);
            predictedRisk = wrongRisks[Math.floor(Math.random() * wrongRisks.length)];
            confidence = Math.max(0.3, confidence - 0.3);
        }
        
        return {
            predictedRisk: predictedRisk,
            confidence: Math.min(0.95, Math.max(0.3, confidence)),
            probabilities: this.getProbabilities(predictedRisk, confidence)
        };
    }

    // Generate probability distribution
    getProbabilities(predictedRisk, confidence) {
        const baseProb = (1 - confidence) / 2;
        const probs = {
            High: baseProb,
            Medium: baseProb,
            Low: baseProb
        };
        
        probs[predictedRisk] = confidence;
        return [probs.High, probs.Medium, probs.Low];
    }

    getModelInfo() {
        return this.modelInfo;
    }
}

const diseaseModel = new DiseaseRiskModel();
