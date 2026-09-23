class VibrationCalc {
    constructor() {
        this.data = [];
        this.peaks = [];
        this.equilibriumPos = 0;
        this.dampingRatio = 0;
        this.dampingCoeff = 0;
        this.logDecay = 0;
        this.period = 0;
        this.stopTime = 0;
        this.dampedModel = null;
    }

    setData(data, peaks) {
        this.data = data;
        this.peaks = peaks;
    }

    setDampedModel(model) {
        this.dampedModel = model;
    }

    calculateEquilibrium() {
        if (this.data.length === 0) {
            this.equilibriumPos = 0;
            return 0;
        }

        const lastQuarter = this.data.slice(Math.floor(this.data.length * 3 / 4));
        const yValues = lastQuarter.map(d => d.y);
        this.equilibriumPos = math.mean(yValues);
        
        return this.equilibriumPos;
    }

    calculateLogDecay() {
        if (this.dampedModel && this.dampedModel.lambda > 0) {
            const modelPeriod = this.period > 0 ? this.period : this.dampedModel.period;
            this.logDecay = Math.abs(this.dampedModel.lambda * modelPeriod);
            return this.logDecay;
        }

        if (this.peaks.length < 2) {
            this.logDecay = 0;
            return 0;
        }

        const decays = [];

        for (let i = 0; i < this.peaks.length - 1; i++) {
            const amplitude1 = Math.abs(this.peaks[i].y - this.equilibriumPos);
            const amplitude2 = Math.abs(this.peaks[i + 1].y - this.equilibriumPos);
            
            if (amplitude1 > 1e-9 && amplitude2 > 1e-9) {
                const value = Math.log(amplitude1 / amplitude2);
                if (isFinite(value)) decays.push(value);
            }
        }

        decays.sort((a, b) => a - b);
        const mid = Math.floor(decays.length / 2);
        const median = decays.length === 0
            ? 0
            : decays.length % 2
                ? decays[mid]
                : (decays[mid - 1] + decays[mid]) / 2;
        // 阻尼衰减率是幅值，检测噪声导致的符号反转不应产生负阻尼。
        this.logDecay = Math.abs(median);
        return this.logDecay;
    }

    calculateDampingRatio() {
        if (this.logDecay === 0) {
            this.dampingRatio = 0;
            return 0;
        }

        const logDecay = Math.abs(this.logDecay);
        this.dampingRatio = logDecay / Math.sqrt(Math.pow(2 * Math.PI, 2) + Math.pow(logDecay, 2));
        return this.dampingRatio;
    }

    calculateDampingCoefficient(mass = 1) {
        if (this.period === 0 || this.dampingRatio === 0) {
            this.dampingCoeff = 0;
            return 0;
        }

        const omegaN = 2 * Math.PI / this.period;
        this.dampingCoeff = 2 * mass * this.dampingRatio * omegaN;
        return this.dampingCoeff;
    }

    calculateStopTime(threshold = 0.01) {
        if (this.dampedModel && this.dampedModel.lambda > 0 && this.dampedModel.amplitude > 0) {
            const effectiveThreshold = Math.max(threshold, this.dampedModel.amplitude * 0.02);
            const duration = Math.log(this.dampedModel.amplitude / effectiveThreshold) / this.dampedModel.lambda;
            this.stopTime = this.dampedModel.t0 + Math.max(0, duration);
            return this.stopTime;
        }

        if (this.peaks.length === 0 || this.dampingRatio === 0) {
            this.stopTime = 0;
            return 0;
        }

        const firstPeak = this.peaks[0];
        const initialAmplitude = Math.abs(firstPeak.y - this.equilibriumPos);
        
        if (initialAmplitude === 0) {
            this.stopTime = 0;
            return 0;
        }

        const omegaN = 2 * Math.PI / (this.period || 1);
        const omegaD = omegaN * Math.sqrt(1 - Math.pow(this.dampingRatio, 2));
        
        const timeToStop = (Math.log(initialAmplitude / threshold)) / (this.dampingRatio * omegaN);
        
        const lastTime = this.data.length > 0 ? this.data[this.data.length - 1].timestamp : 0;
        this.stopTime = Math.max(lastTime, timeToStop);
        
        return this.stopTime;
    }

    calculateAll(period) {
        this.period = period;
        
        this.calculateEquilibrium();
        this.calculateLogDecay();
        this.calculateDampingRatio();
        this.calculateDampingCoefficient();
        this.calculateStopTime();

        return {
            equilibriumPos: this.equilibriumPos,
            dampingRatio: this.dampingRatio,
            dampingCoeff: this.dampingCoeff,
            logDecay: this.logDecay,
            period: this.period,
            stopTime: this.stopTime
        };
    }

    getDisplacementAtTime(data, time) {
        if (data.length === 0) return 0;

        const idx = data.findIndex(d => d.timestamp >= time);
        
        if (idx === 0) return data[0].y;
        if (idx === -1) return data[data.length - 1].y;

        const prev = data[idx - 1];
        const curr = data[idx];
        const t = (time - prev.timestamp) / (curr.timestamp - prev.timestamp);
        
        return prev.y + t * (curr.y - prev.y);
    }

    getEquilibriumPos() {
        return this.equilibriumPos;
    }

    getDampingRatio() {
        return this.dampingRatio;
    }

    getDampingCoeff() {
        return this.dampingCoeff;
    }

    getLogDecay() {
        return this.logDecay;
    }

    getPeriod() {
        return this.period;
    }

    getStopTime() {
        return this.stopTime;
    }

    reset() {
        this.data = [];
        this.peaks = [];
        this.equilibriumPos = 0;
        this.dampingRatio = 0;
        this.dampingCoeff = 0;
        this.logDecay = 0;
        this.period = 0;
        this.stopTime = 0;
        this.dampedModel = null;
    }
}
