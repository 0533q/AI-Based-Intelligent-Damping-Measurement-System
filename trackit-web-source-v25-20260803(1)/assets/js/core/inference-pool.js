/**
 * 推理 Worker 池：N 个 Web Worker 并行跑 ORT 推理。
 * - submit(): 有空闲 worker 则立即提交并返回 Promise，全部忙碌返回 null（容量信号）
 * - 帧像素通过 transferable ArrayBuffer 零拷贝送入 Worker
 * - 结果乱序到达，由调用方按时间戳排序
 */
class InferenceWorkerPool {
    /**
     * @param {Object} options
     *   workerCount - worker 数量（受硬件线程数与内存限制）
     *   modelPath / modelOpts - ORT 会话参数
     */
    constructor({ workerCount = 3, modelPath, modelOpts } = {}) {
        this.workerCount = Math.max(1, workerCount);
        this.modelPath = modelPath;
        this.modelOpts = modelOpts;
        this.workers = [];
        this.busy = new Set();
        this.pending = new Map();
        this._id = 0;
        this.disposed = false;
        this.cancelled = false;
        this._readyResolve = null;
        this.ready = new Promise((resolve) => { this._readyResolve = resolve; });
        this._readyCount = 0;
        this._readyOk = 0;
        this.initFailed = false;
    }

    /** 创建 worker 并并行预加载模型会话 */
    async init() {
        for (let i = 0; i < this.workerCount; i++) {
            try {
                const w = new Worker('assets/js/core/inference-worker.js');
                w.onmessage = (e) => this._onMessage(e.data);
                w.onerror = (e) => this._onWorkerError(e);
                this.workers.push(w);
                this._sendTo(w, { id: ++this._id, type: 'warmup', modelPath: this.modelPath, modelOpts: this.modelOpts });
            } catch (e) {
                console.error('Worker 创建失败:', e);
            }
        }
        if (this.workers.length === 0) {
            this._readyResolve();
            this.initFailed = true;
            return this;
        }
        await this.ready;
        this.initFailed = this._readyOk === 0;
        return this;
    }

    _sendTo(worker, msg) {
        try {
            worker.postMessage(msg);
        } catch (e) {
            console.error('Worker 消息发送失败:', e);
        }
    }

    _onMessage(data) {
        if (data.type === 'warmup-result') {
            this._readyCount++;
            if (data.ok) {
                this._readyOk++;
            } else {
                console.error('Worker 会话预加载失败:', data.error);
            }
            if (this._readyCount >= this.workers.length && this._readyResolve) {
                this._readyResolve();
            }
            return;
        }
        const entry = this.pending.get(data.id);
        if (!entry) return;
        this.pending.delete(data.id);
        this.busy.delete(entry.workerIndex);
        if (data.ok) {
            entry.resolve(data.detections);
        } else {
            entry.reject(new Error(data.error || 'Worker 推理失败'));
        }
    }

    _onWorkerError(e) {
        console.error('Worker 错误:', e.message || e);
        // 该 worker 的挂起任务全部失败
        const dead = this.workers.indexOf(e.target);
        if (dead >= 0) this.busy.delete(dead);
    }

    hasCapacity() {
        return !this.disposed && !this.cancelled && this.busy.size < this.workers.length;
    }

    /**
     * 提交一帧。返回 Promise<detections>；无空闲 worker 时返回 null。
     * imageData 的 data.buffer 将被转移（零拷贝），提交后不可再使用。
     */
    submit(imageData, mapping) {
        if (!this.hasCapacity()) return null;
        const workerIndex = this._findFreeWorker();
        if (workerIndex === -1) return null;

        const id = ++this._id;
        const promise = new Promise((resolve, reject) => {
            this.pending.set(id, { resolve, reject, workerIndex });
        });
        this.busy.add(workerIndex);
        this._sendTo(this.workers[workerIndex], {
            id,
            type: 'infer',
            modelPath: this.modelPath,
            modelOpts: this.modelOpts,
            imageData: {
                buf: imageData.data.buffer,
                width: imageData.width,
                height: imageData.height
            },
            mapping
        });
        return promise;
    }

    _findFreeWorker() {
        for (let i = 0; i < this.workers.length; i++) {
            if (!this.busy.has(i)) return i;
        }
        return -1;
    }

    /** 等待所有已提交任务完成 */
    async drain() {
        while (this.pending.size > 0) {
            await new Promise(r => setTimeout(r, 50));
        }
    }

    cancel() {
        this.cancelled = true;
        for (const entry of this.pending.values()) {
            entry.reject(new Error('分析已取消'));
        }
        this.pending.clear();
        this.busy.clear();
    }

    dispose() {
        this.disposed = true;
        for (const w of this.workers) {
            try { w.terminate(); } catch (e) {}
        }
        this.workers = [];
        this.pending.clear();
        this.busy.clear();
    }
}
