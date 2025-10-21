class Visualization {
    static confidenceChart = null;
    static trainingChart = null;

    static drawConfidenceChart(probabilities, labels) {
        const ctx = document.getElementById('confidenceChart');
        if (!ctx) {
            console.warn('Confidence chart canvas not found');
            return;
        }
        
        if (this.confidenceChart) {
            this.confidenceChart.destroy();
        }

        const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'];

        try {
            this.confidenceChart = new Chart(ctx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Classification Confidence',
                        data: probabilities.map(p => p * 100),
                        backgroundColor: colors,
                        borderColor: colors.map(color => this.darkenColor(color, 20)),
                        borderWidth: 2,
                        borderRadius: 5
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Confidence (%)'
                            },
                            grid: {
                                color: 'rgba(0,0,0,0.1)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Biological Class'
                            },
                            grid: {
                                display: false
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Confidence: ${context.raw.toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error drawing confidence chart:', error);
        }
    }

    static drawTrainingHistory(history) {
        if (!history || !history.acc || !history.val_acc) {
            console.warn('No training history data available');
            return;
        }

        console.log('Training history available for visualization:', history);
    }

    static drawFeatureImportance(featureImportance, featureNames) {
        console.log('Feature importance visualization:', {
            importance: featureImportance,
            names: featureNames
        });
    }

    static darkenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = Math.max(0, (num >> 16) - amt);
        const G = Math.max(0, (num >> 8 & 0x00FF) - amt);
        const B = Math.max(0, (num & 0x0000FF) - amt);
        return "#" + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1);
    }

    static clearCharts() {
        if (this.confidenceChart) {
            this.confidenceChart.destroy();
            this.confidenceChart = null;
        }
        if (this.trainingChart) {
            this.trainingChart.destroy();
            this.trainingChart = null;
        }
    }
}
