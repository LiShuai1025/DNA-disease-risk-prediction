class Visualization {
    static displayConfidenceChart(predictions) {
        const ctx = document.getElementById('confidence-chart').getContext('2d');
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['人类', '细菌', '病毒', '植物'],
                datasets: [{
                    label: '置信度',
                    data: Array.from(predictions),
                    backgroundColor: [
                        '#3498db', '#2ecc71', '#e74c3c', '#9b59b6'
                    ]
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }

    static displayAlignment(sequence, prediction) {
        const alignmentViewer = document.getElementById('alignment-viewer');
        
        // 简单的序列比对可视化
        const topPrediction = this.getTopPrediction(prediction);
        alignmentViewer.innerHTML = `
            <div class="alignment-result">
                <h4>最可能来源: ${topPrediction.label} (${(topPrediction.confidence * 100).toFixed(2)}%)</h4>
                <div class="sequence-preview">${sequence.substring(0, 100)}...</div>
                <div class="alignment-stats">
                    <p>序列长度: ${sequence.length} bp</p>
                    <p>GC含量: ${this.calculateGCContent(sequence).toFixed(2)}%</p>
                </div>
            </div>
        `;
    }

    static getTopPrediction(predictions) {
        const maxIndex = predictions.indexOf(Math.max(...predictions));
        return {
            label: ['人类', '细菌', '病毒', '植物'][maxIndex],
            confidence: predictions[maxIndex]
        };
    }

    static calculateGCContent(sequence) {
        const gcCount = (sequence.match(/[GC]/gi) || []).length;
        return (gcCount / sequence.length) * 100;
    }
}
