class ChartRender {
    constructor() {
        this.vibrationChart = null;
        this.lossChart = null;
        this.vibrationCtx = null;
        this.lossCtx = null;
    }

    initCharts() {
        // 暗色主题全局默认
        Chart.defaults.color = '#8b9bb4';
        Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.12)';
        Chart.defaults.font.family = '"PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif';
        Chart.defaults.font.size = 11;

        this.vibrationCtx = document.getElementById('vibrationChart').getContext('2d');
        this.lossCtx = document.getElementById('lossChart').getContext('2d');

        this.vibrationChart = new Chart(this.vibrationCtx, {
            type: 'line',
            data: { datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { intersect: false, mode: 'index' },
                animation: { duration: 300 },
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: '时间 (s)', color: '#8b9bb4' },
                        grid: { color: 'rgba(148, 163, 184, 0.08)' },
                        ticks: { color: '#8b9bb4' }
                    },
                    y: {
                        title: { display: true, text: '位移 (cm)', color: '#8b9bb4' },
                        grid: { color: 'rgba(148, 163, 184, 0.08)' },
                        ticks: { color: '#8b9bb4' }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#cbd5e1', usePointStyle: true, pointStyle: 'circle', boxWidth: 6, boxHeight: 6, padding: 16 }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 16, 28, 0.92)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(148, 163, 184, 0.25)',
                        borderWidth: 1,
                        callbacks: {
                            label: (context) => {
                                return `(${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                            }
                        }
                    }
                }
            }
        });

        this.lossChart = new Chart(this.lossCtx, {
            type: 'line',
            data: { labels: [], datasets: [] },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Epoch', color: '#8b9bb4' },
                        grid: { color: 'rgba(148, 163, 184, 0.08)' },
                        ticks: { color: '#8b9bb4' }
                    },
                    y: {
                        title: { display: true, text: 'Loss', color: '#8b9bb4' },
                        grid: { color: 'rgba(148, 163, 184, 0.08)' },
                        ticks: { color: '#8b9bb4' },
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#cbd5e1', usePointStyle: true, pointStyle: 'circle', boxWidth: 6, boxHeight: 6, padding: 16 }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 16, 28, 0.92)',
                        titleColor: '#e2e8f0',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(148, 163, 184, 0.25)',
                        borderWidth: 1
                    }
                }
            }
        });
    }

    renderVibrationChart(data, fittedData, extrapolatedData, equilibriumPos, stopTime, period = 0) {
        const datasets = [];
        const clip = true;

        if (data.length > 0) {
            datasets.push({
                label: '原始数据',
                data: data.map(d => ({ x: d.timestamp, y: d.y })),
                backgroundColor: 'rgba(56, 189, 248, 0.75)',
                borderColor: 'rgba(56, 189, 248, 1)',
                pointRadius: 3,
                pointHoverRadius: 5,
                showLine: false,
                clip
            });
        }

        if (fittedData.length > 0) {
            datasets.push({
                label: '拟合曲线',
                data: fittedData.map(d => ({ x: d.timestamp, y: d.y })),
                borderColor: 'rgba(245, 158, 11, 1)',
                backgroundColor: 'rgba(245, 158, 11, 0.08)',
                borderWidth: 2.5,
                pointRadius: 0,
                tension: 0.4,
                fill: false,
                clip
            });
        }

        if (extrapolatedData.length > 0) {
            const extrapPoints = extrapolatedData.filter(d => d.extrapolated);
            if (extrapPoints.length > 0) {
                datasets.push({
                    label: '预测曲线',
                    data: extrapPoints.map(d => ({ x: d.timestamp, y: d.y })),
                    borderColor: 'rgba(45, 212, 191, 1)',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 0,
                    tension: 0.4,
                    fill: false,
                    clip
                });
            }
        }

        if (equilibriumPos !== undefined && equilibriumPos !== null && data.length > 0) {
            const xValues = data.map(d => d.timestamp);
            const minX = Math.min(...xValues);
            const maxX = Math.max(...xValues);

            datasets.push({
                label: '平衡位置',
                data: [
                    { x: minX, y: equilibriumPos },
                    { x: maxX, y: equilibriumPos }
                ],
                borderColor: 'rgba(163, 230, 53, 1)',
                borderWidth: 1.5,
                borderDash: [4, 4],
                pointRadius: 0,
                fill: false,
                clip
            });
        }

        const fullXMax = Math.max(
            data.length ? Math.max(...data.map(d => d.timestamp)) : 0,
            extrapolatedData.length ? Math.max(...extrapolatedData.map(d => d.timestamp)) : 0
        );

        // y 轴：按原始数据 2%~98% 分位设定范围，离群点不再撑爆坐标轴
        let yMin = -0.5, yMax = 0.5;
        if (data.length >= 4) {
            const ys = data.map(d => d.y).sort((a, b) => a - b);
            const q = (p) => ys[Math.min(ys.length - 1, Math.floor(ys.length * p))];
            const lo = q(0.02), hi = q(0.98);
            const pad = (hi - lo) * 0.15 || 0.5;
            yMin = lo - pad;
            yMax = hi + pad;
        }

        // 停止时间使用贯穿绘图区的红色虚线，并在平衡位置绘制醒目的星形点。
        if (stopTime > 0 && equilibriumPos !== undefined && stopTime <= fullXMax + 1e-6) {
            datasets.push({
                label: '停止时间',
                data: [
                    { x: stopTime, y: yMin },
                    { x: stopTime, y: equilibriumPos },
                    { x: stopTime, y: yMax }
                ],
                borderColor: 'rgba(255, 77, 79, 0.9)',
                backgroundColor: 'rgba(255, 77, 79, 1)',
                borderWidth: 1.5,
                borderDash: [5, 4],
                pointRadius: context => context.dataIndex === 1 ? 9 : 0,
                pointHoverRadius: context => context.dataIndex === 1 ? 11 : 0,
                pointStyle: 'star',
                showLine: true,
                tension: 0,
                fill: false,
                clip: false
            });
        }

        // x 轴默认窗口：完整时间范围（0 → 数据/预测末端），波形铺满全宽、刻度正常。
        // 「聚焦」按钮可切换为前几个周期的放大视图。
        const dataXMax = data.length ? Math.max(...data.map(d => d.timestamp)) : 0;
        const dataXMin = data.length ? Math.min(...data.map(d => d.timestamp)) : 0;
        const xPadding = fullXMax > 0
            ? Math.max((fullXMax - Math.min(0, dataXMin)) * 0.05, period > 0 ? period * 0.5 : 0.5)
            : 0;
        const displayXMax = (fullXMax || dataXMax) + xPadding;
        this._dataXMax = dataXMax;
        this._fullXMax = displayXMax;
        this._defaultXMin = 0;
        this._defaultXMax = displayXMax;
        this._activeRange = this._computeActiveRange(data, equilibriumPos, period, dataXMin, dataXMax);

        this.vibrationChart.options.scales.x.min = this._currentXMin !== undefined ? this._currentXMin : this._defaultXMin;
        this.vibrationChart.options.scales.x.max = this._currentXMax !== undefined ? this._currentXMax : this._defaultXMax;
        this.vibrationChart.options.scales.y.min = yMin;
        this.vibrationChart.options.scales.y.max = yMax;

        this.vibrationChart.data.datasets = datasets;
        this.vibrationChart.update('none');
    }

    /**
     * 计算聚焦窗口 [start, end]：幅度显著（> 峰值 15%）的振动起点，
     * 起点前移半周期、至少覆盖 3 个完整周期（放大查看用）。
     */
    _computeActiveRange(data, equilibrium, period, xMin, xMax) {
        const span = xMax - xMin;
        if (data.length < 4 || span <= 0) return [xMin, xMax];

        const eq = equilibrium ?? data[0].y;
        const amps = data.map(d => Math.abs(d.y - eq));
        const maxAmp = Math.max(...amps);
        const p = period > 0 ? period : span / 8;
        const minWindow = Math.max(3 * p, 1.5);
        if (maxAmp < 1e-6) return [xMin, Math.min(xMax, xMin + minWindow)];

        const th = maxAmp * 0.15;
        const first = data.findIndex(d => Math.abs(d.y - eq) > th);
        if (first < 0) return [xMin, Math.min(xMax, xMin + minWindow)];

        const start = Math.max(xMin, data[first].timestamp - p / 2);
        return [start, Math.min(xMax, start + minWindow)];
    }

    /** 图表 x 轴窗口切换（聚焦区间 / 全览） */
    setXRange(min, max) {
        this._currentXMin = min;
        this._currentXMax = max;
        if (this.vibrationChart) {
            this.vibrationChart.options.scales.x.min = min;
            this.vibrationChart.options.scales.x.max = max;
            this.vibrationChart.update('none');
        }
    }

    /** 恢复默认（全范围）窗口 */
    resetXRange() {
        this._currentXMin = undefined;
        this._currentXMax = undefined;
        if (this.vibrationChart) {
            this.vibrationChart.options.scales.x.min = this._defaultXMin;
            this.vibrationChart.options.scales.x.max = this._defaultXMax;
            this.vibrationChart.update('none');
        }
    }

    /** 聚焦窗口：有效振动区间的开头几波（放大查看用） */
    focusActiveRange() {
        const [min, max] = this._activeRange;
        this.setXRange(min, max);
    }

    getDefaultXMax() {
        return this._defaultXMax;
    }

    getDefaultXMin() {
        return this._defaultXMin;
    }

    getActiveRange() {
        return this._activeRange;
    }

    getFullXMax() {
        return this._fullXMax;
    }

    renderLossChart(lossHistory, valLossHistory) {
        const labels = lossHistory.map((_, i) => i + 1);
        const datasets = [];

        if (lossHistory.length > 0) {
            datasets.push({
                label: '训练Loss',
                data: lossHistory,
                borderColor: 'rgba(56, 189, 248, 1)',
                backgroundColor: 'rgba(56, 189, 248, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            });
        }

        if (valLossHistory.length > 0) {
            datasets.push({
                label: '验证Loss',
                data: valLossHistory,
                borderColor: 'rgba(251, 113, 133, 1)',
                backgroundColor: 'rgba(251, 113, 133, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            });
        }

        this.lossChart.data.labels = labels;
        this.lossChart.data.datasets = datasets;
        this.lossChart.update('none');
    }

    updateLossChart(epoch, loss, valLoss) {
        const labels = this.lossChart.data.labels || [];
        const trainLoss = this.lossChart.data.datasets[0]?.data || [];
        const valLossData = this.lossChart.data.datasets[1]?.data || [];

        labels.push(epoch + 1);
        trainLoss.push(loss);
        valLossData.push(valLoss);

        if (!this.lossChart.data.datasets[0]) {
            this.lossChart.data.datasets.push({
                label: '训练Loss',
                data: trainLoss,
                borderColor: 'rgba(56, 189, 248, 1)',
                backgroundColor: 'rgba(56, 189, 248, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            });
        }

        if (!this.lossChart.data.datasets[1]) {
            this.lossChart.data.datasets.push({
                label: '验证Loss',
                data: valLossData,
                borderColor: 'rgba(251, 113, 133, 1)',
                backgroundColor: 'rgba(251, 113, 133, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            });
        }

        this.lossChart.update('none');
    }

    clearCharts() {
        if (this.vibrationChart) {
            this.vibrationChart.data.datasets = [];
            this.vibrationChart.update();
        }

        if (this.lossChart) {
            this.lossChart.data.labels = [];
            this.lossChart.data.datasets = [];
            this.lossChart.update();
        }
    }

    dispose() {
        if (this.vibrationChart) {
            this.vibrationChart.destroy();
        }
        if (this.lossChart) {
            this.lossChart.destroy();
        }
    }
}
