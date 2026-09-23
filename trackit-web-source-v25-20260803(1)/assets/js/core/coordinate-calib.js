class CoordinateCalib {
    constructor() {
        this.origin = { x: 0, y: 0 };
        this.scale = 1;
        this.rulerWidthPx = 0;
        this.rulerWidthCm = 5;
        this.isCalibrated = false;
    }

    calibrate(firstFrameDetections) {
        const pointer = firstFrameDetections.find(d => d.class === 'red_pointer');
        const ruler = firstFrameDetections.find(d => d.class === 'green_scale_5cm');

        if (!pointer) {
            throw new Error('第一帧未检测到指针');
        }

        this.origin = { x: pointer.x, y: pointer.y };

        if (ruler) {
            this.rulerWidthPx = ruler.width;
            this.scale = this.rulerWidthCm / this.rulerWidthPx;
        } else {
            this.scale = 0.01;
        }

        this.isCalibrated = true;

        return {
            origin: this.origin,
            scale: this.scale,
            rulerWidthPx: this.rulerWidthPx,
            rulerWidthCm: this.rulerWidthCm
        };
    }

    pixelToPhysical(px, py) {
        if (!this.isCalibrated) {
            throw new Error('未进行标定');
        }

        const x = (px - this.origin.x) * this.scale;
        const y = (this.origin.y - py) * this.scale;

        return { x, y };
    }

    physicalToPixel(px, py) {
        if (!this.isCalibrated) {
            throw new Error('未进行标定');
        }

        const x = px / this.scale + this.origin.x;
        const y = this.origin.y - py / this.scale;

        return { x, y };
    }

    convertDetectionToPhysical(detection) {
        return this.pixelToPhysical(detection.x, detection.y);
    }

    convertFrameData(frameData) {
        const result = [];
        let previous = null;
        let velocity = { x: 0, y: 0 };

        [...frameData].sort((a, b) => a.timestamp - b.timestamp).forEach(frame => {
            const candidates = frame.detections
                .filter(d => d.class === 'red_pointer')
                .sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
            if (candidates.length === 0) return;

            let pointer = candidates[0];
            if (previous) {
                const dt = Math.max(1e-3, frame.timestamp - previous.timestamp);
                const predicted = {
                    x: previous.x + velocity.x * dt,
                    y: previous.y + velocity.y * dt
                };
                const baseGate = Math.max(
                    24,
                    this.rulerWidthPx > 0 ? this.rulerWidthPx * 0.35 : 0,
                    (previous.width || pointer.width || 8) * 4
                );
                const gate = baseGate * Math.max(1, Math.min(4, dt / 0.1));

                pointer = candidates.reduce((best, candidate) => {
                    const distance = Math.hypot(candidate.x - predicted.x, candidate.y - predicted.y);
                    const score = distance / gate - (candidate.confidence || 0) * 0.3;
                    return !best || score < best.score ? { ...candidate, distance, score } : best;
                }, null);

                // 低置信度且远离预测轨迹的检测视为跳点，交给后续插值补齐。
                if (pointer.distance > gate && (pointer.confidence || 0) < 0.8) return;

                const measuredVelocity = {
                    x: (pointer.x - previous.x) / dt,
                    y: (pointer.y - previous.y) / dt
                };
                velocity.x = velocity.x * 0.65 + measuredVelocity.x * 0.35;
                velocity.y = velocity.y * 0.65 + measuredVelocity.y * 0.35;
            }

            const physical = this.pixelToPhysical(pointer.x, pointer.y);
            result.push({
                timestamp: frame.timestamp,
                frameIndex: frame.frameIndex,
                x: physical.x,
                y: physical.y,
                rawX: pointer.x,
                rawY: pointer.y,
                confidence: pointer.confidence || 0
            });
            previous = {
                timestamp: frame.timestamp,
                x: pointer.x,
                y: pointer.y,
                width: pointer.width
            };
        });
        
        return result;
    }

    getOrigin() {
        return this.origin;
    }

    getScale() {
        return this.scale;
    }

    isCalibrated() {
        return this.isCalibrated;
    }

    reset() {
        this.origin = { x: 0, y: 0 };
        this.scale = 1;
        this.rulerWidthPx = 0;
        this.isCalibrated = false;
    }
}
