class DataProcessor {
    constructor() {
        this.rawData = [];
        this.cleanedData = [];
        this.normalizedData = [];
        this.peaks = [];
        this.period = 0;
        this.fitData = [];
        this.vibrationStartTime = 0;
    }

    setRawData(data) {
        this.rawData = data;
    }

    apply3SigmaFilter(data) {
        if (data.length < 3) return data;

        const values = data.map(d => d.y);
        const mean = math.mean(values);
        const std = math.std(values);
        const lowerBound = mean - 3 * std;
        const upperBound = mean + 3 * std;

        return data.filter(d => d.y >= lowerBound && d.y <= upperBound);
    }

    _median(values) {
        if (values.length === 0) return 0;
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2
            ? sorted[mid]
            : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    /**
     * 局部 Hampel/MAD 异常点修复。
     * 与全局 3σ 删除相比，它不会把阻尼振动开头的正常大振幅误判为异常；
     * 检出的孤立跳点使用局部中位数替换，从而保留原始时间轴。
     */
    applyHampelFilter(data, windowRadius = 4, threshold = 3.5) {
        if (data.length < windowRadius * 2 + 1) return [...data];

        return data.map((point, index) => {
            const start = Math.max(0, index - windowRadius);
            const end = Math.min(data.length, index + windowRadius + 1);
            const window = data.slice(start, end).map(d => d.y);
            const median = this._median(window);
            const mad = this._median(window.map(value => Math.abs(value - median)));
            const robustSigma = 1.4826 * mad;

            if (robustSigma > 1e-9 && Math.abs(point.y - median) > threshold * robustSigma) {
                return { ...point, y: median };
            }
            return { ...point };
        });
    }

    /** 去除明显低置信度检测；没有置信度字段时保持兼容。 */
    filterByConfidence(data) {
        const confidences = data
            .map(d => d.confidence)
            .filter(value => typeof value === 'number' && value > 0)
            .sort((a, b) => a - b);
        if (confidences.length < Math.max(8, data.length * 0.5)) return [...data];

        const q20 = confidences[Math.floor(confidences.length * 0.2)];
        const threshold = Math.max(0.28, q20);
        const filtered = data.filter(d => !(d.confidence > 0) || d.confidence >= threshold);
        return filtered.length >= 8 ? filtered : [...data];
    }

    linearInterpolation(data) {
        if (data.length < 2) return data;

        const interpolated = [];
        const timeDiffs = [];
        for (let i = 1; i < data.length; i++) {
            const diff = data[i].timestamp - data[i - 1].timestamp;
            if (diff > 0 && isFinite(diff)) timeDiffs.push(diff);
        }
        const expectedStep = this._median(timeDiffs);
        if (!(expectedStep > 0)) return [...data];
        
        for (let i = 0; i < data.length - 1; i++) {
            interpolated.push(data[i]);
            
            const current = data[i];
            const next = data[i + 1];
            const timeDiff = next.timestamp - current.timestamp;
            
            if (timeDiff > expectedStep * 1.5) {
                const steps = Math.max(2, Math.round(timeDiff / expectedStep));
                const stepTime = timeDiff / steps;
                const stepX = (next.x - current.x) / steps;
                const stepY = (next.y - current.y) / steps;
                
                for (let j = 1; j < steps; j++) {
                    interpolated.push({
                        timestamp: current.timestamp + j * stepTime,
                        x: current.x + j * stepX,
                        y: current.y + j * stepY,
                        confidence: Math.min(current.confidence || 0, next.confidence || 0),
                        interpolated: true
                    });
                }
            }
        }
        
        if (data.length > 0) {
            interpolated.push(data[data.length - 1]);
        }
        
        return interpolated;
    }

    correctLensDistortion(data) {
        const corrected = data.map(d => {
            const k1 = -0.15;
            const r2 = d.x * d.x + d.y * d.y;
            const factor = 1 + k1 * r2;
            
            return {
                timestamp: d.timestamp,
                x: d.x * factor,
                y: d.y * factor
            };
        });
        
        return corrected;
    }

    /**
     * 自动寻找从静止到显著振动的切换点。
     * 基线取视频开头约 1 秒，连续窗口内多数点越过稳健阈值才视为开始。
     */
    detectVibrationStart(data) {
        if (data.length < 12) return 0;
        const dt = this._median(data.slice(1).map((d, i) => d.timestamp - data[i].timestamp).filter(v => v > 0));
        const baselineCount = Math.max(6, Math.min(Math.floor(data.length * 0.15), Math.round(1 / (dt || 0.1))));
        const baseline = data.slice(0, baselineCount).map(d => d.y);
        const baselineMedian = this._median(baseline);
        const baselineMad = this._median(baseline.map(value => Math.abs(value - baselineMedian)));
        const robustSigma = 1.4826 * baselineMad;
        const allY = data.map(d => d.y);
        const globalRange = Math.max(...allY) - Math.min(...allY);
        const baselineRange = Math.max(...baseline) - Math.min(...baseline);

        // 视频开头已经在振动时，不强行裁掉前段。
        if (globalRange <= 1e-9 || baselineRange > globalRange * 0.35) return 0;

        const threshold = Math.max(0.015, robustSigma * 6, globalRange * 0.08);
        const windowSize = 5;
        for (let i = baselineCount; i <= data.length - windowSize; i++) {
            let activeCount = 0;
            for (let j = 0; j < windowSize; j++) {
                if (Math.abs(data[i + j].y - baselineMedian) > threshold) activeCount++;
            }
            if (activeCount >= 3) return Math.max(0, i - 1);
        }
        return 0;
    }

    /** 5 点二次 Savitzky-Golay 平滑，抑制检测抖动同时尽量保留振幅与相位。 */
    smoothSignal(data) {
        if (data.length < 5) return data.map(point => ({ ...point }));
        const coefficients = [-3, 12, 17, 12, -3];
        return data.map((point, index) => {
            if (index < 2 || index > data.length - 3) return { ...point };
            let y = 0;
            for (let j = -2; j <= 2; j++) y += data[index + j].y * coefficients[j + 2];
            return { ...point, y: y / 35, smoothed: true };
        });
    }

    normalize(data) {
        if (data.length === 0) return [];

        const yValues = data.map(d => d.y);
        const minY = Math.min(...yValues);
        const maxY = Math.max(...yValues);
        const range = maxY - minY || 1;

        return data.map(d => ({
            timestamp: d.timestamp,
            x: d.x,
            y: ((d.y - minY) / range) * 2 - 1,
            confidence: d.confidence
        }));
    }

    /** 使用自相关寻找越过负相关后的第一个正相关峰，避免噪声小尖峰缩短周期。 */
    estimatePeriod(data) {
        if (data.length < 12) return 0;
        const diffs = [];
        for (let i = 1; i < data.length; i++) {
            const diff = data[i].timestamp - data[i - 1].timestamp;
            if (diff > 0 && isFinite(diff)) diffs.push(diff);
        }
        const dt = this._median(diffs);
        if (!(dt > 0)) return 0;

        const mean = data.reduce((sum, d) => sum + d.y, 0) / data.length;
        const centered = data.map(d => d.y - mean);
        const maxLag = Math.min(Math.floor(data.length / 2), 600);
        const correlations = new Float64Array(maxLag + 1);

        for (let lag = 2; lag <= maxLag; lag++) {
            let numerator = 0, leftEnergy = 0, rightEnergy = 0;
            for (let i = lag; i < data.length; i++) {
                const a = centered[i];
                const b = centered[i - lag];
                numerator += a * b;
                leftEnergy += a * a;
                rightEnergy += b * b;
            }
            correlations[lag] = numerator / (Math.sqrt(leftEnergy * rightEnergy) || 1);
        }

        let crossedNegative = false;
        for (let lag = 3; lag < maxLag; lag++) {
            if (correlations[lag] < 0) crossedNegative = true;
            if (crossedNegative && correlations[lag] > 0.05 &&
                correlations[lag] > correlations[lag - 1] &&
                correlations[lag] >= correlations[lag + 1]) {
                return lag * dt;
            }
        }
        return 0;
    }

    findPeaks(data, period = 0) {
        if (data.length < 3) return [];

        const smoothed = data.map((point, index) => {
            let sum = 0, count = 0;
            for (let j = Math.max(0, index - 2); j <= Math.min(data.length - 1, index + 2); j++) {
                sum += data[j].y;
                count++;
            }
            return sum / count;
        });
        const candidates = [];
        for (let i = 1; i < data.length - 1; i++) {
            if (smoothed[i] > smoothed[i - 1] && smoothed[i] >= smoothed[i + 1]) {
                candidates.push({
                    timestamp: data[i].timestamp,
                    y: data[i].y,
                    index: i
                });
            }
        }

        const peaks = [];
        const minSeparation = period > 0 ? period * 0.6 : 0;
        for (const candidate of candidates) {
            const previous = peaks[peaks.length - 1];
            if (!previous || candidate.timestamp - previous.timestamp >= minSeparation) {
                peaks.push(candidate);
            } else if (candidate.y > previous.y) {
                peaks[peaks.length - 1] = candidate;
            }
        }

        this.peaks = peaks;
        return peaks;
    }

    calculatePeriod(peaks) {
        if (peaks.length < 2) return 0;

        let totalDiff = 0;
        let count = 0;

        for (let i = 1; i < peaks.length; i++) {
            totalDiff += peaks[i].timestamp - peaks[i - 1].timestamp;
            count++;
        }

        this.period = count > 0 ? totalDiff / count : 0;
        return this.period;
    }

    process() {
        if (this.rawData.length === 0) {
            throw new Error('无原始数据');
        }

        let data = this.rawData
            .filter(d => isFinite(d.timestamp) && isFinite(d.x) && isFinite(d.y))
            .sort((a, b) => a.timestamp - b.timestamp);

        data = this.filterByConfidence(data);
        data = this.applyHampelFilter(data, 5, 3.0);
        data = this.linearInterpolation(data);
        
        this.cleanedData = data;
        
        const normalized = this.normalize(data);
        this.normalizedData = normalized;
        
        const startIndex = this.detectVibrationStart(data);
        this.vibrationStartTime = data[startIndex]?.timestamp || data[0].timestamp;
        this.fitData = this.smoothSignal(data.slice(startIndex));

        const autocorrelationPeriod = this.estimatePeriod(this.fitData);
        const peaks = this.findPeaks(this.fitData, autocorrelationPeriod);
        this.period = autocorrelationPeriod || this.calculatePeriod(peaks);

        return {
            cleanedData: this.cleanedData,
            fitData: this.fitData,
            normalizedData: this.normalizedData,
            peaks: this.peaks,
            period: this.period,
            vibrationStartTime: this.vibrationStartTime
        };
    }

    getRawData() {
        return this.rawData;
    }

    getCleanedData() {
        return this.cleanedData;
    }

    getNormalizedData() {
        return this.normalizedData;
    }

    getPeaks() {
        return this.peaks;
    }

    getPeriod() {
        return this.period;
    }

    reset() {
        this.rawData = [];
        this.cleanedData = [];
        this.normalizedData = [];
        this.peaks = [];
        this.period = 0;
        this.fitData = [];
        this.vibrationStartTime = 0;
    }
}
