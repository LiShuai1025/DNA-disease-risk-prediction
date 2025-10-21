class Visualization {
    static createConfidenceChart(predictions, classNames) {
        const ctx = document.getElementById('confidence-chart').getContext('2d');
        
        // Destroy previous chart if exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        const colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'];
        
        this.chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: classNames,
                datasets: [{
                    label: 'Confidence',
                    data: predictions.map(p => (p * 100).toFixed(2)),
                    backgroundColor: colors,
                    borderColor: colors.map(color => this.darkenColor(color, 20)),
                    borderWidth: 2,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.raw}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Confidence (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Biological Source'
                        }
                    }
                }
            }
        });
    }

    static displaySequenceAnalysis(sequence, prediction, confidence) {
        const alignmentElement = document.getElementById('alignment-viewer');
        const cleanSequence = sequence.toUpperCase().replace(/[^ATCG]/g, '');
        
        // Create a simple sequence visualization
        const preview = cleanSequence.length > 100 ? 
            cleanSequence.substring(0, 100) + '...' : 
            cleanSequence;
        
        alignmentElement.innerHTML = `
            <div class="analysis-result">
                <div class="analysis-header">
                    <h4>Sequence Analysis</h4>
                    <span class="confidence-badge">${confidence}% confidence</span>
                </div>
                <div class="sequence-preview">
                    <strong>Sequence Preview:</strong>
                    <div class="sequence-text">${preview}</div>
                </div>
                <div class="analysis-details">
                    <p><strong>Predicted Source:</strong> <span class="prediction-highlight ${prediction.toLowerCase()}">${prediction}</span></p>
                    <p><strong>Sequence Length:</strong> ${cleanSequence.length} base pairs</p>
                    <p><strong>Analysis:</strong> This sequence exhibits characteristics typical of ${prediction.toLowerCase()} DNA.</p>
                </div>
            </div>
        `;
    }

    static darkenColor(color, percent) {
        const num = parseInt(color.slice(1), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }

    static createSequenceVisualization(sequence) {
        // Create a colored sequence visualization
        const sequenceElement = document.createElement('div');
        sequenceElement.className = 'sequence-visualization';
        
        let html = '';
        for (let i = 0; i < Math.min(sequence.length, 200); i++) {
            const base = sequence[i];
            let color = '#666';
            
            switch(base) {
                case 'A': color = '#FF6B6B'; break;
                case 'T': color = '#4ECDC4'; break;
                case 'C': color = '#45B7D1'; break;
                case 'G': color = '#96CEB4'; break;
            }
            
            html += `<span style="color: ${color}; font-weight: bold;">${base}</span>`;
            
            if ((i + 1) % 10 === 0) {
                html += ' ';
            }
        }
        
        sequenceElement.innerHTML = html;
        return sequenceElement;
    }
}
