class Visualization {
    static confidenceChart = null;

    static drawConfidenceChart(probabilities, labels) {
        const ctx = document.getElementById('confidenceChart').getContext('2d');
        
        // Destroy existing chart if any
        if (this.confidenceChart) {
            this.confidenceChart.destroy();
        }

        const colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'];

        this.confidenceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Classification Confidence',
                    data: probabilities.map(p => p * 100),
                    backgroundColor: colors,
                    borderColor: colors.map(color => this.darkenColor(color, 20)),
                    borderWidth: 1
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
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Biological Class'
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
    }

    static darkenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;
        return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
            (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
            (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
    }
}
 static drawTrainingHistory(history) {
        // 训练历史可视化
        const ctx = document.createElement('canvas');
        ctx.width = 400;
        ctx.height = 300;
        
        // 这里可以添加训练损失和准确率的图表
        // 由于复杂度，这里省略具体实现
        return ctx;
    }

    static drawFeatureImportance(featureImportance, featureNames) {
        // 特征重要性可视化
        const ctx = document.createElement('canvas');
        ctx.width = 400;
        ctx.height = 300;
        
        // 实现特征重要性条形图
        return ctx;
    }
}
