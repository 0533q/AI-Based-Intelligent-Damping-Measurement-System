class VideoProcessor {
    constructor() {
        this.video = null;
        this.videoUrl = null;
        this.fps = 30;
        this.duration = 0;
        this.frameCount = 0;
        this.videoWidth = 0;
        this.videoHeight = 0;
        this.SAMPLE_INTERVAL = 3;
        this.cancelled = false;
        this._canvases = new Map();
        this._contexts = new Map();
    }

    loadVideo(file) {
        return new Promise((resolve, reject) => {
            if (file.type !== 'video/mp4') {
                reject(new Error('仅支持MP4格式视频'));
                return;
            }

            const url = URL.createObjectURL(file);
            const video = document.createElement('video');
            video.src = url;
            video.crossOrigin = 'anonymous';
            video.playsInline = true;

            video.onloadedmetadata = () => {
                if (video.duration > 60) {
                    URL.revokeObjectURL(url);
                    reject(new Error('视频时长不能超过60秒'));
                    return;
                }

                let videoFps = 30;
                if (video.videoTracks && video.videoTracks[0] &&
                    video.videoTracks[0].getSettings &&
                    video.videoTracks[0].getSettings().frameRate) {
                    videoFps = video.videoTracks[0].getSettings().frameRate;
                }
                // 关键钳制：部分视频（尤其手机可变帧率录制）getSettings().frameRate
                // 会返回异常大值（几百~上千），导致采样时间步长被压缩到 ~0.003s，
                // 所有采样点堆在视频开头 → 曲线在 t=0 挤成竖线
                if (!(typeof videoFps === 'number' && isFinite(videoFps))) videoFps = 30;
                videoFps = Math.min(120, Math.max(1, videoFps));

                this.video = video;
                this.videoUrl = url;
                this.fps = videoFps;
                this.duration = video.duration;
                this.frameCount = Math.floor(video.duration * videoFps);
                this.videoWidth = video.videoWidth;
                this.videoHeight = video.videoHeight;
                this.cancelled = false;

                resolve({
                    frameCount: this.frameCount,
                    fps: this.fps,
                    duration: this.duration,
                    width: this.videoWidth,
                    height: this.videoHeight,
                    sampleInterval: this.SAMPLE_INTERVAL,
                    sampledFrames: this.getSampledFrameCount()
                });
            };

            video.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('视频加载失败'));
            };

            video.load();
        });
    }

    _getCanvas(targetSize) {
        if (!this._canvases.has(targetSize)) {
            const canvas = document.createElement('canvas');
            canvas.width = targetSize;
            canvas.height = targetSize;
            this._canvases.set(targetSize, canvas);
            this._contexts.set(targetSize, canvas.getContext('2d', { willReadFrequently: true }));
        }
        return { canvas: this._canvases.get(targetSize), ctx: this._contexts.get(targetSize) };
    }

    /**
     * 将视频当前帧 letterbox 绘制到 targetSize×targetSize 画布。
     * 返回 ImageData + 坐标映射（模型坐标 → 视频像素坐标）。
     */
    captureCurrentFrame(targetSize) {
        const { ctx } = this._getCanvas(targetSize);
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, targetSize, targetSize);

        const scale = Math.min(targetSize / this.videoWidth, targetSize / this.videoHeight);
        const drawW = Math.round(this.videoWidth * scale);
        const drawH = Math.round(this.videoHeight * scale);
        const padLeft = Math.floor((targetSize - drawW) / 2);
        const padTop = Math.floor((targetSize - drawH) / 2);

        ctx.drawImage(this.video, padLeft, padTop, drawW, drawH);

        return {
            imageData: ctx.getImageData(0, 0, targetSize, targetSize),
            mapping: { scale, padLeft, padTop, videoWidth: this.videoWidth, videoHeight: this.videoHeight }
        };
    }

    /**
     * 流式抓帧：以正常速度播放视频，每 sampleInterval 帧采样一次，
     * 采样时暂停视频并等待 onFrame 完成推理，随后继续播放。
     * 相比逐帧 seek：零 seek 开销，提取耗时 ≈ 视频时长，且不存储任何像素。
     *
     * @param {Object} options
     *   targetSize - 抓帧 letterbox 目标尺寸（必须与模型输入一致）
     *   sampleInterval - 采样间隔（帧）
     *   onFrame(frame, meta) - 每采样帧回调，frame = {imageData, mapping}
     *   onProgress(sampled, total, videoProgress01)
     *   signal - { cancelled: boolean } 取消信号（app 层共享）
     * @returns {Promise<{sampledFrames: number, duration: number}>}
     */
    async captureWithSampling({ targetSize = 1280, sampleInterval = this.SAMPLE_INTERVAL, onFrame, onProgress, signal } = {}) {
        return new Promise((resolve, reject) => {
            const video = this.video;
            if (!video) {
                reject(new Error('视频未加载'));
                return;
            }

            const fps = this.fps || 30;
            const totalExpected = Math.floor(this.duration * fps / sampleInterval) + 1;
            const nextSampleTime = () => (sampledCount * sampleInterval) / fps;

            let sampledCount = 0;
            let busy = false;
            let resuming = false;   // play() 未 settle 期间禁止再次 pause（避免中断 play 请求）
            let done = false;
            let rafId = 0;

            const isCancelled = () => this.cancelled || (signal && signal.cancelled);
            const nearEnd = () => video.ended || video.currentTime >= this.duration - 1e-3;

            const finish = (err) => {
                if (done) return;
                done = true;
                cancelAnimationFrame(rafId);
                video.pause();
                video.muted = false;
                if (err) reject(err);
                else resolve({ sampledFrames: sampledCount, duration: this.duration });
            };

            /** 恢复播放：play() 的 promise 完成前不允许新的采样（避免 pause 中断播放请求） */
            const resumePlayback = () => {
                resuming = true;
                const p = video.play();
                if (p && typeof p.finally === 'function') {
                    p.finally(() => { resuming = false; }).catch(() => {});
                } else {
                    resuming = false;
                }
            };

            const tick = () => {
                if (done) return;
                if (isCancelled()) { finish(); return; }

                if (!busy && !resuming) {
                    if (nearEnd() || sampledCount >= totalExpected) {
                        finish();
                        return;
                    }
                    if (video.currentTime >= nextSampleTime() - 1e-3) {
                        sampledCount++;
                        busy = true;
                        video.pause();

                        let frame;
                        try {
                            frame = this.captureCurrentFrame(targetSize);
                        } catch (e) {
                            console.error('帧提取失败:', e);
                            busy = false;
                            resumePlayback();
                            rafId = requestAnimationFrame(tick);
                            return;
                        }

                        const timestamp = video.currentTime;
                        Promise.resolve()
                            .then(() => onFrame(frame, { timestamp, frameIndex: sampledCount - 1 }))
                            .catch(err => console.error(`帧处理失败: ${err.message}`))
                            .finally(() => {
                                busy = false;
                                if (onProgress) onProgress(sampledCount, totalExpected, Math.min(1, video.currentTime / this.duration));
                                if (done) return;
                                if (nearEnd()) { finish(); return; }
                                resumePlayback();
                            });
                    }
                }
                rafId = requestAnimationFrame(tick);
            };

            // 先建立播放，再启动采样循环
            video.muted = true;
            video.currentTime = 0;
            const playPromise = video.play();
            if (playPromise && typeof playPromise.then === 'function') {
                playPromise.then(() => {
                    rafId = requestAnimationFrame(tick);
                }).catch(err => {
                    if (!done) finish(new Error(`视频无法播放: ${err?.message || err}`));
                });
            } else {
                rafId = requestAnimationFrame(tick);
            }
        });
    }

    /**
     * Worker 池流水线模式：视频连续播放（不暂停），按采样时间点抓帧提交给
     * submit()；submit 返回 null 表示池忙，此时不推进采样点（自动降速到
     * 池可持续的采样率，保持时间覆盖均匀）。视频播完后等待池排空由调用方负责。
     *
     * 自适应播放速率：测量池实际吞吐，将 video.playbackRate 调整到
     * 「吞吐 = 采样率」的平衡点。池越快播放越接近 1×（速度与精度兼得）；
     * 池慢则放慢播放，保证采样密度不下降（精度不损失）。
     *
     * @param {Object} options
     *   targetSize - 抓帧 letterbox 目标尺寸
     *   sampleInterval - 最小采样间隔（帧），实际采样率受池吞吐限制
     *   submit(frame, meta) - 返回 Promise 或 null（null=池忙，放弃该采样点）
     *   onProgress(sampled, total, videoProgress01)
     *   signal - 取消信号
     */
    async captureWithWorkerPool({ targetSize = 1280, sampleInterval = this.SAMPLE_INTERVAL, submit, onProgress, signal } = {}) {
        return new Promise((resolve, reject) => {
            const video = this.video;
            if (!video) {
                reject(new Error('视频未加载'));
                return;
            }

            const fps = this.fps || 30;
            // 期望采样率上限 20Hz：fps 虚高时避免播放速率被压到地板值
            const desiredWallRate = Math.min(fps / sampleInterval, 20);
            const totalExpected = Math.floor(this.duration * fps / sampleInterval) + 1;
            let sampledCount = 0;
            let nextSampleTime = 0;
            let done = false;
            let rafId = 0;
            const t0 = performance.now();
            let playbackRate = 1;

            const isCancelled = () => this.cancelled || (signal && signal.cancelled);
            const nearEnd = () => video.ended || video.currentTime >= this.duration - 1e-3;

            const finish = (err) => {
                if (done) return;
                done = true;
                cancelAnimationFrame(rafId);
                video.muted = false;
                if (err) reject(err);
                else resolve({ sampledFrames: sampledCount, duration: this.duration });
            };

            /** 按实测吞吐调整播放速率，使提交率≈采样率 */
            const adaptPlaybackRate = () => {
                const elapsedSec = (performance.now() - t0) / 1000;
                if (elapsedSec < 3 || sampledCount < 6) return;
                const wallRate = sampledCount / elapsedSec;
                if (wallRate <= 0) return;
                const target = Math.min(1, (wallRate / desiredWallRate) * 0.85);
                const next = Math.max(0.15, Math.min(1, (playbackRate + target) / 2));
                if (Math.abs(next - playbackRate) > 0.02) {
                    playbackRate = next;
                    video.playbackRate = playbackRate;
                }
            };

            const tick = () => {
                if (done) return;
                if (isCancelled()) { finish(); return; }
                if (nearEnd()) {
                    if (onProgress) onProgress(sampledCount, totalExpected, 1);
                    finish();
                    return;
                }
                if (video.currentTime >= nextSampleTime - 1e-3) {
                    let frame;
                    try {
                        frame = this.captureCurrentFrame(targetSize);
                    } catch (e) {
                        console.error('帧提取失败:', e);
                        rafId = requestAnimationFrame(tick);
                        return;
                    }
                    const accepted = submit(frame, { timestamp: video.currentTime, frameIndex: sampledCount });
                    if (accepted) {
                        sampledCount++;
                        nextSampleTime = (sampledCount * sampleInterval) / fps;
                        adaptPlaybackRate();
                        if (onProgress) onProgress(sampledCount, totalExpected, Math.min(1, video.currentTime / this.duration));
                    }
                    // 池忙：不推进采样点，下个 tick 重试同一采样点（视频继续播放）
                }
                rafId = requestAnimationFrame(tick);
            };

            video.muted = true;
            video.currentTime = 0;
            video.playbackRate = 1;
            const playPromise = video.play();
            if (playPromise && typeof playPromise.then === 'function') {
                playPromise.then(() => {
                    rafId = requestAnimationFrame(tick);
                }).catch(err => {
                    if (!done) finish(new Error(`视频无法播放: ${err?.message || err}`));
                });
            } else {
                rafId = requestAnimationFrame(tick);
            }
        });
    }

    cleanup() {
        if (this.videoUrl) {
            URL.revokeObjectURL(this.videoUrl);
        }
        this.video = null;
        this.videoUrl = null;
        this.cancelled = false;
    }

    cancel() {
        this.cancelled = true;
    }

    getFrameCount() {
        return this.frameCount;
    }

    getFPS() {
        return this.fps;
    }

    getDuration() {
        return this.duration;
    }

    getSampledFrameCount() {
        return Math.floor(this.frameCount / this.SAMPLE_INTERVAL) + 1;
    }

    getSampleInterval() {
        return this.SAMPLE_INTERVAL;
    }
}
