/**
 * 纯 JS 实现的 3→32→16→1 MLP + Adam 训练器。
 * 替代 TensorFlow.js：网络极小（约 650 参数），200 epoch 在任意手机上 1 秒内完成，
 * 且移除 tfjs 的 CDN 依赖，离线环境可用。公共 API 与原 tfjs 版本保持一致。
 */
class NNFitter {
    constructor() {
        this.history = { loss: [], valLoss: [] };
        this.isTrained = false;
        this.params = { epochs: 200, batchSize: 16, learningRate: 0.001 };
        this._W = null;
        this._adam = null;
        this._beta1 = 0.9;
        this._beta2 = 0.999;
        this._eps = 1e-8;
        this._dampedModel = null;
    }

    _build() {
        const dims = [3, 32, 16, 1];
        const W = [];
        for (let l = 0; l < dims.length - 1; l++) {
            const fanIn = dims[l], fanOut = dims[l + 1];
            const w = new Float64Array(fanIn * fanOut);
            const limit = Math.sqrt(6 / (fanIn + fanOut));
            for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 - 1) * limit;
            W.push(w, new Float64Array(fanOut));
        }
        this._W = W;
        this._adam = W.map(w => ({ m: new Float64Array(w.length), v: new Float64Array(w.length) }));
    }

    /** 前向传播，返回 y */
    _forward(x, h1, h2) {
        const [W1, b1, W2, b2, W3, b3] = this._W;

        for (let j = 0; j < 32; j++) {
            let s = b1[j];
            for (let i = 0; i < 3; i++) s += W1[j * 3 + i] * x[i];
            h1[j] = s > 0 ? s : 0;
        }
        for (let j = 0; j < 16; j++) {
            let s = b2[j];
            for (let i = 0; i < 32; i++) s += W2[j * 32 + i] * h1[i];
            h2[j] = s > 0 ? s : 0;
        }
        let y = b3[0];
        for (let i = 0; i < 16; i++) y += W3[i] * h2[i];
        return y;
    }

    /** 反向传播，累积梯度（MSE loss，2*(y-t) 不除以 batch，更新时再缩放） */
    _backward(x, yTarget, y, h1, h2, grads) {
        const [W1, b1, W2, b2, W3, b3] = this._W;
        const [gW1, gb1, gW2, gb2, gW3, gb3] = grads;

        const d3 = 2 * (y - yTarget);

        for (let i = 0; i < 16; i++) {
            gW3[i] += d3 * h2[i];
        }
        gb3[0] += d3;

        const d2 = new Float64Array(16);
        for (let j = 0; j < 16; j++) {
            if (h2[j] > 0) {
                const dj = d3 * W3[j];
                d2[j] = dj;
                for (let i = 0; i < 32; i++) gW2[j * 32 + i] += dj * h1[i];
                gb2[j] += dj;
            }
        }

        const d1 = new Float64Array(32);
        for (let i = 0; i < 32; i++) {
            if (h1[i] > 0) {
                let di = 0;
                for (let j = 0; j < 16; j++) di += d2[j] * W2[j * 32 + i];
                d1[i] = di;
                for (let k = 0; k < 3; k++) gW1[i * 3 + k] += di * x[k];
                gb1[i] += di;
            }
        }
    }

    /** Adam 参数更新，t 从 1 开始 */
    _applyAdam(grads, t) {
        const lr = this.params.learningRate;
        const b1c = 1 - Math.pow(this._beta1, t);
        const b2c = 1 - Math.pow(this._beta2, t);

        for (let p = 0; p < this._W.length; p++) {
            const w = this._W[p], g = grads[p];
            const { m, v } = this._adam[p];
            for (let i = 0; i < w.length; i++) {
                m[i] = this._beta1 * m[i] + (1 - this._beta1) * g[i];
                v[i] = this._beta2 * v[i] + (1 - this._beta2) * g[i] * g[i];
                w[i] -= lr * (m[i] / b1c) / (Math.sqrt(v[i] / b2c) + this._eps);
            }
        }
    }

    _prepareData(data) {
        const n = data.length;
        const m = n - 3;
        const inputs = new Float64Array(m * 3);
        const outputs = new Float64Array(m);
        for (let i = 3; i < n; i++) {
            const k = i - 3;
            inputs[k * 3] = data[i - 3].y;
            inputs[k * 3 + 1] = data[i - 2].y;
            inputs[k * 3 + 2] = data[i - 1].y;
            outputs[k] = data[i].y;
        }
        return { inputs, outputs, m };
    }

    _median(values) {
        if (values.length === 0) return 0;
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2
            ? sorted[mid]
            : (sorted[mid - 1] + sorted[mid]) / 2;
    }

    _getTimeStep(data) {
        const diffs = [];
        for (let i = 1; i < data.length; i++) {
            const diff = data[i].timestamp - data[i - 1].timestamp;
            if (diff > 0 && isFinite(diff)) diffs.push(diff);
        }
        return this._median(diffs) || 0.033;
    }

    /** 通过自相关估算完整振动周期，优先使用预处理阶段的可靠周期。 */
    _estimatePeriod(data, preferredPeriod = 0) {
        const n = data.length;
        const dt = this._getTimeStep(data);
        const span = data[n - 1].timestamp - data[0].timestamp;

        const mean = data.reduce((sum, d) => sum + d.y, 0) / n;
        const centered = data.map(d => d.y - mean);
        const maxLag = Math.min(Math.floor(n / 2), 600);
        const correlations = new Float64Array(maxLag + 1);

        for (let lag = 2; lag <= maxLag; lag++) {
            let numerator = 0, leftEnergy = 0, rightEnergy = 0;
            for (let i = lag; i < n; i++) {
                const a = centered[i];
                const b = centered[i - lag];
                numerator += a * b;
                leftEnergy += a * a;
                rightEnergy += b * b;
            }
            correlations[lag] = numerator / (Math.sqrt(leftEnergy * rightEnergy) || 1);
        }

        let crossedNegative = false;
        let bestLag = 0;
        for (let lag = 3; lag < maxLag; lag++) {
            if (correlations[lag] < 0) crossedNegative = true;
            if (crossedNegative && correlations[lag] > 0.05 &&
                correlations[lag] > correlations[lag - 1] &&
                correlations[lag] >= correlations[lag + 1]) {
                bestLag = lag;
                break;
            }
        }

        if (!bestLag) {
            let bestCorrelation = -Infinity;
            for (let lag = 4; lag < maxLag; lag++) {
                if (correlations[lag] > correlations[lag - 1] &&
                    correlations[lag] >= correlations[lag + 1] &&
                    correlations[lag] > bestCorrelation) {
                    bestCorrelation = correlations[lag];
                    bestLag = lag;
                }
            }
        }

        const autocorrelationPeriod = bestLag ? bestLag * dt : 0;
        const preferredIsValid = preferredPeriod >= 6 * dt && preferredPeriod <= span / 2;
        if (preferredIsValid && autocorrelationPeriod > 0) {
            const ratio = preferredPeriod / autocorrelationPeriod;
            if (ratio >= 0.75 && ratio <= 1.25) return preferredPeriod;
        }
        if (autocorrelationPeriod > 0) return autocorrelationPeriod;
        return preferredIsValid ? preferredPeriod : Math.max(6 * dt, span / 5);
    }

    /** 从半周期包络的对数斜率估算衰减率 λ。 */
    _estimateDecayRate(data, period) {
        const t0 = data[0].timestamp;
        const tail = data.slice(Math.floor(data.length * 0.75));
        const equilibrium = tail.reduce((sum, d) => sum + d.y, 0) / tail.length;
        const halfPeriod = period / 2;
        const bins = new Map();

        for (const point of data) {
            const bin = Math.floor((point.timestamp - t0) / halfPeriod);
            const amplitude = Math.abs(point.y - equilibrium);
            const previous = bins.get(bin);
            if (!previous || amplitude > previous.amplitude) {
                bins.set(bin, { time: point.timestamp - t0, amplitude });
            }
        }

        const envelope = [...bins.values()];
        const maxAmplitude = Math.max(...envelope.map(p => p.amplitude), 0);
        const usable = envelope.filter(p => p.amplitude > Math.max(1e-9, maxAmplitude * 0.03));
        if (usable.length < 2) return 0;

        const xs = usable.map(p => p.time);
        const ys = usable.map(p => Math.log(p.amplitude));
        const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
        const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;
        let numerator = 0, denominator = 0;
        for (let i = 0; i < xs.length; i++) {
            numerator += (xs[i] - meanX) * (ys[i] - meanY);
            denominator += (xs[i] - meanX) ** 2;
        }
        const slope = denominator > 0 ? numerator / denominator : 0;
        // 检测噪声可能让包络回归斜率短暂为正；阻尼系统只取衰减强度。
        return Math.abs(slope);
    }

    _solve3x3(matrix, vector) {
        const augmented = matrix.map((row, i) => [...row, vector[i]]);
        for (let col = 0; col < 3; col++) {
            let pivot = col;
            for (let row = col + 1; row < 3; row++) {
                if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) pivot = row;
            }
            if (Math.abs(augmented[pivot][col]) < 1e-12) return null;
            [augmented[col], augmented[pivot]] = [augmented[pivot], augmented[col]];
            const divisor = augmented[col][col];
            for (let j = col; j < 4; j++) augmented[col][j] /= divisor;
            for (let row = 0; row < 3; row++) {
                if (row === col) continue;
                const factor = augmented[row][col];
                for (let j = col; j < 4; j++) augmented[row][j] -= factor * augmented[col][j];
            }
        }
        return augmented.map(row => row[3]);
    }

    /**
     * 拟合 y = equilibrium + exp(-λt) * (B cos(ωt) + C sin(ωt))。
     * λ 被限制为正数，因此拟合与外推的振幅包络严格单调递减。
     */
    _fitDampedModel(data, preferredPeriod = 0) {
        if (data.length < 8) return null;

        const t0 = data[0].timestamp;
        const span = data[data.length - 1].timestamp - t0;
        const dt = this._getTimeStep(data);
        if (!(span > 0 && dt > 0)) return null;

        const periodSeed = this._estimatePeriod(data, preferredPeriod);
        const decaySeed = this._estimateDecayRate(data, periodSeed);
        const minPeriod = Math.max(4 * dt, periodSeed * 0.7);
        const maxPeriod = Math.min(span / 2, periodSeed * 1.3);
        if (!(maxPeriod > minPeriod)) return null;

        // 至少保证观测区间衰减 20%，同时用稳健包络估计限制退化解。
        const minDecay = Math.max(Math.log(1.25) / span, decaySeed * 0.5);
        const maxDecay = Math.max(8 / span, decaySeed * 1.75, minDecay * 1.5);
        let best = null;

        for (let pi = 0; pi <= 40; pi++) {
            const period = minPeriod + (maxPeriod - minPeriod) * pi / 40;
            const omega = 2 * Math.PI / period;
            for (let li = 0; li <= 30; li++) {
                const lambda = minDecay + (maxDecay - minDecay) * li / 30;
                let s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
                let b0 = 0, b1 = 0, b2 = 0;

                for (const point of data) {
                    const t = point.timestamp - t0;
                    const decay = Math.exp(-lambda * t);
                    const p = decay * Math.cos(omega * t);
                    const q = decay * Math.sin(omega * t);
                    s01 += p;
                    s02 += q;
                    s11 += p * p;
                    s12 += p * q;
                    s22 += q * q;
                    b0 += point.y;
                    b1 += p * point.y;
                    b2 += q * point.y;
                }

                const coefficients = this._solve3x3(
                    [[data.length, s01, s02], [s01, s11, s12], [s02, s12, s22]],
                    [b0, b1, b2]
                );
                if (!coefficients) continue;

                const [equilibrium, cosCoefficient, sinCoefficient] = coefficients;
                let error = 0;
                for (const point of data) {
                    const t = point.timestamp - t0;
                    const predicted = equilibrium + Math.exp(-lambda * t) * (
                        cosCoefficient * Math.cos(omega * t) +
                        sinCoefficient * Math.sin(omega * t)
                    );
                    error += (predicted - point.y) ** 2;
                }
                error /= data.length;

                if (!best || error < best.error) {
                    best = {
                        t0,
                        equilibrium,
                        cosCoefficient,
                        sinCoefficient,
                        amplitude: Math.hypot(cosCoefficient, sinCoefficient),
                        lambda,
                        omega,
                        period,
                        error
                    };
                }
            }
        }

        if (best) {
            // 最小二乘容易被大量靠近平衡位置的散点拉成小振幅。
            // 使用前 25% 数据的高分位包络修正初始振幅，降低欠拟合。
            const earlyEnd = Math.max(4, Math.floor(data.length * 0.25));
            const earlyAmplitudes = data.slice(0, earlyEnd)
                .map(point => Math.abs(point.y - best.equilibrium))
                .sort((a, b) => a - b);
            const robustAmplitude = earlyAmplitudes[
                Math.min(earlyAmplitudes.length - 1, Math.floor(earlyAmplitudes.length * 0.85))
            ];
            if (robustAmplitude > best.amplitude && best.amplitude > 1e-9) {
                const scale = robustAmplitude / best.amplitude;
                best.cosCoefficient *= scale;
                best.sinCoefficient *= scale;
                best.amplitude = robustAmplitude;
            }
        }

        return best;
    }

    _predictDampedAt(timestamp) {
        const model = this._dampedModel;
        if (!model) return null;
        const t = timestamp - model.t0;
        return model.equilibrium + Math.exp(-model.lambda * Math.max(0, t)) * (
            model.cosCoefficient * Math.cos(model.omega * t) +
            model.sinCoefficient * Math.sin(model.omega * t)
        );
    }

    async train(data, onEpochEnd, preferredPeriod = 0) {
        if (data.length < 4) {
            throw new Error('数据量不足，无法训练');
        }

        if (!this._W) this._build();

        const { inputs, outputs, m } = this._prepareData(data);
        const splitIndex = Math.floor(m * 0.8);
        if (splitIndex < 4) {
            throw new Error('数据量不足，无法训练');
        }

        const idx = new Int32Array(splitIndex);
        for (let i = 0; i < splitIndex; i++) idx[i] = i;

        const h1 = new Float64Array(32);
        const h2 = new Float64Array(16);
        const B = this.params.batchSize;
        this.history = { loss: [], valLoss: [] };

        for (let epoch = 0; epoch < this.params.epochs; epoch++) {
            for (let i = splitIndex - 1; i > 0; i--) {
                const j = (Math.random() * (i + 1)) | 0;
                const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            }

            const grads = this._W.map(w => new Float64Array(w.length));
            let trainLoss = 0;

            for (let s = 0; s < splitIndex; s++) {
                const k = idx[s];
                const x = inputs.subarray(k * 3, k * 3 + 3);
                const yTarget = outputs[k];
                const y = this._forward(x, h1, h2);
                trainLoss += (y - yTarget) * (y - yTarget);
                this._backward(x, yTarget, y, h1, h2, grads);
            }

            for (const g of grads) {
                for (let i = 0; i < g.length; i++) g[i] /= splitIndex;
            }
            this._applyAdam(grads, epoch + 1);
            trainLoss /= splitIndex;

            let valLoss = 0;
            for (let k = splitIndex; k < m; k++) {
                const x = inputs.subarray(k * 3, k * 3 + 3);
                const yTarget = outputs[k];
                const y = this._forward(x, h1, h2);
                const d = y - yTarget;
                valLoss += d * d;
            }
            valLoss = m > splitIndex ? valLoss / (m - splitIndex) : 0;

            this.history.loss.push(trainLoss);
            this.history.valLoss.push(valLoss);

            if (onEpochEnd) {
                onEpochEnd(epoch, { loss: trainLoss, val_loss: valLoss });
            }

            // 周期性让出主线程，保持 UI 与 Loss 图表实时刷新
            if (epoch % 10 === 9) {
                await new Promise(r => setTimeout(r, 0));
            }
        }

        this._dampedModel = this._fitDampedModel(data, preferredPeriod);
        this.isTrained = true;
        return this.history;
    }

    _predictFromValues(v0, v1, v2) {
        const x = new Float64Array([v0, v1, v2]);
        const h1 = new Float64Array(32);
        const h2 = new Float64Array(16);
        return this._forward(x, h1, h2);
    }

    predictNext(data) {
        if (!this.isTrained || data.length < 3) {
            throw new Error('模型未训练或数据不足');
        }
        return this._predictFromValues(
            data[data.length - 3].y,
            data[data.length - 2].y,
            data[data.length - 1].y
        );
    }

    extrapolate(data, steps) {
        if (!this.isTrained) {
            throw new Error('模型未训练');
        }

        const extrapolated = [...data];
        const timeStep = this._getTimeStep(data);

        for (let i = 0; i < steps; i++) {
            const lastTime = extrapolated[extrapolated.length - 1].timestamp;
            const timestamp = lastTime + timeStep;
            const dampedY = this._predictDampedAt(timestamp);
            const nextY = dampedY === null
                ? this.predictNext(extrapolated.slice(-3))
                : dampedY;
            extrapolated.push({ timestamp, y: nextY, extrapolated: true });
        }

        return extrapolated;
    }

    /** 在固定点数上预测到指定时间，确保很远的停止点也能进入图表范围。 */
    extrapolateToTime(data, endTime, maxPoints = 1200) {
        if (!this.isTrained) throw new Error('模型未训练');
        const extrapolated = [...data];
        const lastTime = data[data.length - 1].timestamp;
        if (!(endTime > lastTime)) return extrapolated;

        const naturalStep = this._getTimeStep(data);
        const steps = Math.max(1, Math.min(maxPoints, Math.ceil((endTime - lastTime) / naturalStep)));
        const step = (endTime - lastTime) / steps;
        for (let i = 1; i <= steps; i++) {
            const timestamp = lastTime + step * i;
            const dampedY = this._predictDampedAt(timestamp);
            const nextY = dampedY === null
                ? this.predictNext(extrapolated.slice(-3))
                : dampedY;
            extrapolated.push({ timestamp, y: nextY, extrapolated: true });
        }
        return extrapolated;
    }

    predictAtTime(data, targetTime) {
        if (!this.isTrained) {
            throw new Error('模型未训练');
        }

        const lastTime = data[data.length - 1].timestamp;

        if (targetTime <= lastTime) {
            const idx = data.findIndex(d => d.timestamp >= targetTime);
            if (idx > 0) {
                const prev = data[idx - 1];
                const curr = data[idx];
                const t = (targetTime - prev.timestamp) / (curr.timestamp - prev.timestamp);
                return prev.y + t * (curr.y - prev.y);
            }
            return data[0].y;
        }

        const dampedY = this._predictDampedAt(targetTime);
        if (dampedY !== null) return dampedY;

        const timeStep = this._getTimeStep(data);
        const steps = Math.ceil((targetTime - lastTime) / timeStep);
        const extrapolated = this.extrapolate(data, steps);
        const idx = extrapolated.findIndex(d => d.timestamp >= targetTime);

        if (idx > 0) {
            const prev = extrapolated[idx - 1];
            const curr = extrapolated[idx];
            const t = (targetTime - prev.timestamp) / (curr.timestamp - prev.timestamp);
            return prev.y + t * (curr.y - prev.y);
        }

        return extrapolated[extrapolated.length - 1].y;
    }

    getHistory() {
        return this.history;
    }

    getDampedModel() {
        return this._dampedModel ? { ...this._dampedModel } : null;
    }

    /** 标准决定系数 R²；1 表示完全拟合，0 表示不优于均值模型。 */
    calculateRSquared(data, fittedData = null) {
        const fitted = fittedData || this.generateFittedData(data);
        const count = Math.min(data.length, fitted.length);
        if (count === 0) return 0;

        let mean = 0;
        for (let i = 0; i < count; i++) mean += data[i].y;
        mean /= count;

        let residualSum = 0;
        let totalSum = 0;
        for (let i = 0; i < count; i++) {
            residualSum += (data[i].y - fitted[i].y) ** 2;
            totalSum += (data[i].y - mean) ** 2;
        }
        if (totalSum < 1e-12) return residualSum < 1e-12 ? 1 : 0;
        return 1 - residualSum / totalSum;
    }

    generateFittedData(data) {
        if (!this.isTrained || data.length < 4) {
            return [];
        }

        const fittedData = [];
        if (this._dampedModel) {
            const model = this._dampedModel;
            const physicalValues = [];
            const learnedValues = [];
            for (let i = 0; i < data.length; i++) {
                const point = data[i];
                const physicalY = this._predictDampedAt(point.timestamp);
                let boundedLearnedY = physicalY;
                if (i >= 3) {
                    const learnedY = this._predictFromValues(
                        data[i - 3].y,
                        data[i - 2].y,
                        data[i - 1].y
                    );
                    const t = Math.max(0, point.timestamp - model.t0);
                    const envelope = model.amplitude * Math.exp(-model.lambda * t);
                    const boundedDeviation = Math.max(-envelope, Math.min(envelope, learnedY - model.equilibrium));
                    boundedLearnedY = model.equilibrium + boundedDeviation;
                }
                physicalValues.push(physicalY);
                learnedValues.push(boundedLearnedY);
            }

            // 在 [0,1] 内求物理曲线与受包络约束的 MLP 曲线的最优混合权重。
            // 数据规则时保持物理曲线；散点较多时自动提高数据驱动成分，避免欠拟合。
            let numerator = 0, denominator = 0;
            for (let i = 0; i < data.length; i++) {
                const delta = learnedValues[i] - physicalValues[i];
                numerator += (data[i].y - physicalValues[i]) * delta;
                denominator += delta * delta;
            }
            const learnedWeight = denominator > 1e-12
                ? Math.max(0, Math.min(1, numerator / denominator))
                : 0;

            for (let i = 0; i < data.length; i++) {
                fittedData.push({
                    timestamp: data[i].timestamp,
                    y: physicalValues[i] + learnedWeight * (learnedValues[i] - physicalValues[i])
                });
            }
            return fittedData;
        }

        for (let i = 3; i < data.length; i++) {
            fittedData.push({
                timestamp: data[i].timestamp,
                y: this._predictFromValues(data[i - 3].y, data[i - 2].y, data[i - 1].y)
            });
        }
        return fittedData;
    }

    /** 训练状态：isTrained 属性（布尔）为公共 API，无需方法包装 */

    dispose() {
        this._W = null;
        this._adam = null;
        this._dampedModel = null;
    }

    reset() {
        this.dispose();
        this.history = { loss: [], valLoss: [] };
        this.isTrained = false;
    }
}
