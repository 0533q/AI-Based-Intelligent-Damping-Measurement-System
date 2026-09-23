class VibrationApp {
    constructor() {
        this.videoProcessor = new VideoProcessor();
        this.yoloDetector = new YoloDetector();
        this.coordinateCalib = new CoordinateCalib();
        this.dataProcessor = new DataProcessor();
        this.nnFitter = new NNFitter();
        this.vibrationCalc = new VibrationCalc();
        this.chartRender = new ChartRender();

        this.videoFile = null;
        this.detections = [];
        this.physicalData = [];
        this.processedData = {};
        this.vibrationParams = {};
        this.fitRSquared = null;
        this.detectionStats = { skipped: 0, upgraded: 0 };
        this.cancelSignal = { cancelled: false };
        this.inferencePool = null;

        this.initUI();
        this.chartRender.initCharts();
    }

    initUI() {
        this.videoInput = document.getElementById('videoInput');
        this.videoPreview = document.getElementById('videoPreview');
        this.detectionCanvas = document.getElementById('detectionCanvas');
        this.detectionCtx = this.detectionCanvas.getContext('2d');
        this.uploadArea = document.getElementById('uploadArea');
        this.videoInfo = document.getElementById('videoInfo');
        this.statusBadge = document.getElementById('statusBadge');

        this.btnStart = document.getElementById('btnStart');
        this.btnReset = document.getElementById('btnReset');
        this.btnExport = document.getElementById('btnExport');

        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');

        this.frameCount = document.getElementById('frameCount');
        this.fpsValue = document.getElementById('fpsValue');
        this.equilibriumPos = document.getElementById('equilibriumPos');
        this.dampingRatio = document.getElementById('dampingRatio');
        this.dampingCoeff = document.getElementById('dampingCoeff');
        this.periodValue = document.getElementById('periodValue');
        this.stopTime = document.getElementById('stopTime');
        this.logDecay = document.getElementById('logDecay');
        this.rSquared = document.getElementById('rSquared');

        this.queryTime = document.getElementById('queryTime');
        this.btnQuery = document.getElementById('btnQuery');
        this.queryResult = document.getElementById('queryResult');

        this.logContainer = document.getElementById('logContainer');
        this.toast = document.getElementById('errorToast');

        this.videoInput.addEventListener('change', (e) => this.handleVideoUpload(e));
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadArea.addEventListener('click', (e) => {
            if (e.target !== this.videoInput) this.videoInput.click();
        });

        this.modeSelect = document.getElementById('modeSelect');

        this.btnStart.addEventListener('click', () => this.startAnalysis());
        this.btnReset.addEventListener('click', () => this.resetApp());
        this.btnExport.addEventListener('click', () => this.exportReport());

        this.btnQuery.addEventListener('click', () => this.queryDisplacement());
        this.queryTime.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.queryDisplacement();
        });

        this.btnViewFirst = document.getElementById('btnViewFirst');
        this.btnViewAll = document.getElementById('btnViewAll');
        this.btnViewFirst.addEventListener('click', () => this.setChartView('first'));
        this.btnViewAll.addEventListener('click', () => this.setChartView('all'));

        this.setStatus('待机', 'idle');
        this.updateStepIndicator(1);
    }

    showError(message) {
        if (!this.toast) return;
        this.toast.textContent = message;
        this.toast.classList.add('show');
        clearTimeout(this._toastTimer);
        this._toastTimer = setTimeout(() => this.toast.classList.remove('show'), 6000);
    }

    setStatus(text, type) {
        this.statusBadge.textContent = text;
        this.statusBadge.className = `status-badge status-${type}`;
    }

    handleVideoUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.loadVideo(file);
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const file = event.dataTransfer.files[0];
        if (file && file.type === 'video/mp4') {
            this.loadVideo(file);
        }
    }

    async loadVideo(file) {
        try {
            this.log('info', `正在加载视频: ${file.name}`);
            this.videoFile = file;
            this.cancelSignal = { cancelled: false };

            const info = await this.videoProcessor.loadVideo(file);

            this.videoPreview.src = URL.createObjectURL(file);
            this.videoPreview.style.display = 'block';
            this.detectionCanvas.width = info.width;
            this.detectionCanvas.height = info.height;
            this.detectionCanvas.style.display = 'block';
            this.detectionCtx.clearRect(0, 0, info.width, info.height);

            this.videoInfo.textContent = `${info.width}×${info.height} · ${info.fps} FPS · ${info.duration.toFixed(2)}s · 共${info.frameCount}帧`;
            this.frameCount.textContent = info.sampledFrames;
            this.fpsValue.textContent = info.fps;

            this.btnStart.disabled = false;
            this.setStatus('就绪', 'ready');
            this.updateStepIndicator(1);
            this.log('success', '视频加载成功');
        } catch (error) {
            this.log('error', `视频加载失败: ${error.message}`);
            this.showError(`视频加载失败: ${error.message}`);
        }
    }

    async startAnalysis() {
        this.btnStart.disabled = true;
        this.btnExport.disabled = true;
        this.cancelSignal = { cancelled: false };

        try {
            await this.step1Detect();
            if (this.cancelSignal.cancelled) return;
            await this.step2Calibrate();
            if (this.cancelSignal.cancelled) return;
            await this.step3CollectData();
            if (this.cancelSignal.cancelled) return;
            await this.step4Preprocess();
            if (this.cancelSignal.cancelled) return;
            await this.step5TrainNN();
            if (this.cancelSignal.cancelled) return;
            await this.step6CalculateVibration();
            if (this.cancelSignal.cancelled) return;
            await this.step7RenderCharts();

            this.btnExport.disabled = false;
            this.updateStepIndicator(8);
            this.setStatus('完成', 'done');
            this.log('success', '分析完成！');
        } catch (error) {
            if (this.cancelSignal.cancelled) return;
            this.log('error', `分析失败: ${error.message}`);
            this.showError(`分析失败: ${error.message}`);
            this.setStatus('失败', 'error');
        } finally {
            this.btnStart.disabled = false;
        }
    }

    /** 步骤1+2（合并）：加载模型 + 流式抓帧识别，抓一帧推理一帧，不存储像素 */
    async step1Detect() {
        this.setStatus('识别中', 'running');
        this.log('info', '加载YOLO模型...');
        this.updateProgress(2, '加载YOLO模型');
        await this.yoloDetector.loadModel();
        this.log('success', 'YOLO模型加载成功');
        this.updateStepIndicator(2);

        this.log('info', '开始流式识别：视频按正常速度播放，多线程并行分析...');
        this.detections = [];
        this.detectionStats = { skipped: 0, upgraded: 0 };

        const usePrecise = this.modeSelect && this.modeSelect.value === 'precise';
        if (this.yoloDetector.adaptive || usePrecise) {
            // 精确模式或自适应（动态输入尺寸）模型走暂停/恢复路径（10Hz 精确采样）
            if (usePrecise) this.log('info', '精确模式：逐帧 10Hz 采样，耗时较长但参数最准');
            await this.detectWithPauseResume();
        } else {
            await this.detectWithWorkerPool();
        }

        if (this.cancelSignal.cancelled) return;

        // 并行结果乱序到达，按时间戳排序后使用
        this.detections.sort((a, b) => a.timestamp - b.timestamp);

        this.log('success', `识别完成: 有效 ${this.detections.length} 帧, 跳过 ${this.detectionStats.skipped} 帧, 升级重检 ${this.detectionStats.upgraded} 帧`);
        if (this.detections.length === 0) {
            throw new Error('未检测到指针，请确保视频中有红色指针');
        }
    }

    /** Worker 池并行推理（固定输入尺寸模型的默认路径） */
    async detectWithWorkerPool() {
        if (!this.inferencePool || this.inferencePool.disposed) {
            try {
                const n = Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 4) - 1));
                this.inferencePool = new InferenceWorkerPool({
                    workerCount: n,
                    // Worker 内相对路径按 Worker 脚本位置解析，必须传绝对 URL
                    modelPath: new URL(this.yoloDetector.modelPath, location.href).href,
                    modelOpts: {
                        executionProviders: ['wasm'],
                        graphOptimizationLevel: 'all',
                        enableCpuMemArena: true,
                        logSeverityLevel: 3
                    }
                });
                this.updateProgress(3, '启动并行推理 Worker...');
                await this.inferencePool.init();
            } catch (e) {
                console.error('Worker 池初始化失败，回退单线程模式:', e);
                if (this.inferencePool) this.inferencePool.dispose();
                this.inferencePool = null;
            }
        }

        if (!this.inferencePool || this.inferencePool.workers.length === 0 || this.inferencePool.initFailed) {
            this.log('warning', '并行推理不可用，回退单线程模式');
            if (this.inferencePool) {
                this.inferencePool.dispose();
                this.inferencePool = null;
            }
            return this.detectWithPauseResume();
        }

        this.log('info', `已启动 ${this.inferencePool.workers.length} 个并行推理 Worker`);
        this.updateProgress(6, `加载模型会话（${this.inferencePool.workers.length} Worker 并行）...`);

        const submits = [];
        await this.videoProcessor.captureWithWorkerPool({
            targetSize: this.yoloDetector.captureTargetSize,
            signal: this.cancelSignal,
            onProgress: (sampled, total, videoProgress) => {
                this.updateProgress(8 + videoProgress * 72, `识别中: 已提交 ${sampled}/${total} 采样帧`);
            },
            submit: (frame, meta) => {
                const p = this.inferencePool.submit(frame.imageData, frame.mapping);
                if (!p) return null;
                submits.push(p.then((detections) => {
                    if (detections.some(d => d.class === 'red_pointer')) {
                        this.detections.push({ timestamp: meta.timestamp, frameIndex: meta.frameIndex, detections });
                    } else {
                        this.detectionStats.skipped++;
                    }
                    this.drawPreviewFrame(detections);
                }).catch((err) => {
                    this.detectionStats.skipped++;
                    console.error(`帧 ${meta.frameIndex} 识别失败: ${err.message}`);
                }));
                return true;
            }
        });

        // 等待所有在途推理完成
        await Promise.allSettled(submits);
    }

    /** 暂停/恢复逐帧路径（自适应模型或 Worker 不可用时回退） */
    async detectWithPauseResume() {
        await this.videoProcessor.captureWithSampling({
            targetSize: this.yoloDetector.captureTargetSize,
            signal: this.cancelSignal,
            onProgress: (sampled, total, videoProgress) => {
                this.updateProgress(8 + videoProgress * 72, `识别中: 已处理 ${sampled}/${total} 采样帧`);
            },
            onFrame: async (frame, meta) => {
                try {
                    let detections = await this.yoloDetector.detect(frame.imageData, frame.mapping);
                    const pointer = detections.find(d => d.class === 'red_pointer');

                    // 自适应模式：未检出指针或置信度低 → 升级 maxInputSize 复查同一帧
                    if (this.yoloDetector.adaptive && (!pointer || pointer.confidence < this.yoloDetector.fastAcceptConfidence)) {
                        const hi = this.videoProcessor.captureCurrentFrame(this.yoloDetector.maxInputSize);
                        const hiDet = await this.yoloDetector.detect(hi.imageData, hi.mapping);
                        const hiPointer = hiDet.find(d => d.class === 'red_pointer');
                        if (hiPointer && (!pointer || hiPointer.confidence > pointer.confidence)) {
                            detections = hiDet;
                        }
                        this.detectionStats.upgraded++;
                    }

                    const hasPointer = detections.some(d => d.class === 'red_pointer');
                    if (hasPointer) {
                        this.detections.push({
                            timestamp: meta.timestamp,
                            frameIndex: meta.frameIndex,
                            detections
                        });
                    } else {
                        this.detectionStats.skipped++;
                    }

                    this.drawPreviewFrame(detections);
                } catch (error) {
                    this.detectionStats.skipped++;
                    console.error(`帧 ${meta.frameIndex} 识别失败: ${error.message}`);
                }
            }
        });
    }

    drawPreviewFrame(detections) {
        const w = this.detectionCanvas.width;
        const h = this.detectionCanvas.height;
        const video = this.videoProcessor.video;
        this.detectionCtx.drawImage(video, 0, 0, w, h);
        this.yoloDetector.drawDetections(this.detectionCtx, detections);
    }

    async step2Calibrate() {
        this.log('info', '开始坐标系标定...');
        this.updateProgress(82, '坐标系标定');
        this.updateStepIndicator(3);

        const firstFrameDetections = this.detections[0]?.detections || [];
        const calibResult = this.coordinateCalib.calibrate(firstFrameDetections);

        this.log('success', `标定完成，原点: (${calibResult.origin.x.toFixed(1)}, ${calibResult.origin.y.toFixed(1)}), 比例尺: ${calibResult.scale.toFixed(4)} cm/px`);
    }

    async step3CollectData() {
        this.log('info', '开始数据采集...');
        this.updateProgress(85, '数据采集');
        this.updateStepIndicator(4);

        this.physicalData = this.coordinateCalib.convertFrameData(this.detections);

        this.log('success', `数据采集完成，共 ${this.physicalData.length} 条数据`);

        if (this.physicalData.length < 10) {
            throw new Error('有效数据不足，请确保视频中指针清晰可见');
        }
    }

    async step4Preprocess() {
        this.log('info', '开始数据预处理...');
        this.updateProgress(87, '数据预处理');

        this.dataProcessor.setRawData(this.physicalData);
        this.processedData = this.dataProcessor.process();

        this.log('success', `预处理完成，清洗后: ${this.processedData.cleanedData.length} 条，有效拟合: ${this.processedData.fitData.length} 条，振动起点: ${this.processedData.vibrationStartTime.toFixed(2)}s，周期: ${this.processedData.period.toFixed(3)}s`);
    }

    async step5TrainNN() {
        this.log('info', '开始神经网络训练（纯JS · 3→32→16→1 · 200轮）...');
        this.updateProgress(88, '神经网络训练');
        this.updateStepIndicator(5);

        await this.nnFitter.train(this.processedData.fitData, (epoch, logs) => {
            this.updateProgress(88 + ((epoch + 1) / 200) * 9, `训练中: Epoch ${epoch + 1}/200, Loss: ${logs.loss.toFixed(6)}`);
            this.chartRender.updateLossChart(epoch, logs.loss, logs.val_loss);
        }, this.processedData.period);

        this.log('success', `神经网络训练完成，最终 Loss: ${this.nnFitter.getHistory().loss[this.nnFitter.getHistory().loss.length - 1]?.toFixed(6)}`);
    }

    async step6CalculateVibration() {
        this.log('info', '开始振动参数计算...');
        this.updateProgress(97, '振动参数计算');
        this.updateStepIndicator(6);

        this.vibrationCalc.setData(this.processedData.fitData, this.processedData.peaks);
        this.vibrationCalc.setDampedModel(this.nnFitter.getDampedModel());
        this.vibrationParams = this.vibrationCalc.calculateAll(this.processedData.period);

        this.equilibriumPos.textContent = this.vibrationParams.equilibriumPos.toFixed(3);
        this.dampingRatio.textContent = this.vibrationParams.dampingRatio.toFixed(4);
        this.dampingCoeff.textContent = this.vibrationParams.dampingCoeff.toFixed(4);
        this.periodValue.textContent = this.vibrationParams.period.toFixed(3);
        this.stopTime.textContent = this.vibrationParams.stopTime.toFixed(2);
        this.logDecay.textContent = this.vibrationParams.logDecay.toFixed(4);

        this.queryTime.disabled = false;
        this.btnQuery.disabled = false;

        this.log('success', `计算完成，平衡位置: ${this.vibrationParams.equilibriumPos.toFixed(3)}cm，阻尼比: ${this.vibrationParams.dampingRatio.toFixed(4)}，停止时间: ${this.vibrationParams.stopTime.toFixed(2)}s`);
    }

    async step7RenderCharts() {
        this.log('info', '开始绘制图表...');
        this.updateProgress(99, '绘制图表');
        this.updateStepIndicator(7);

        const fitData = this.processedData.fitData;
        const fittedData = this.nnFitter.generateFittedData(fitData);
        this.fitRSquared = this.nnFitter.calculateRSquared(fitData, fittedData);
        this.rSquared.textContent = this.fitRSquared.toFixed(4);
        this.vibrationParams.rSquared = this.fitRSquared;
        const cleanedData = this.processedData.cleanedData;
        const lastTime = fitData[fitData.length - 1].timestamp;
        const averageStep = fitData.length > 1
            ? (lastTime - fitData[0].timestamp) / (fitData.length - 1)
            : 0.033;
        const predictionEnd = Math.max(lastTime + averageStep * 100, this.vibrationParams.stopTime);
        const extrapolated = this.nnFitter.extrapolateToTime(fitData, predictionEnd);

        this.chartRender.renderVibrationChart(
            this.processedData.cleanedData,
            fittedData,
            extrapolated,
            this.vibrationParams.equilibriumPos,
            this.vibrationParams.stopTime,
            this.processedData.period
        );

        const history = this.nnFitter.getHistory();
        this.chartRender.renderLossChart(history.loss, history.valLoss);

        // 诊断：原始数据时间范围（排查时间戳坍缩）
        const xs = this.processedData.cleanedData.map(d => d.timestamp);
        if (xs.length > 0) {
            this.log('info', `[诊断] 数据时间范围: [${Math.min(...xs).toFixed(3)}, ${Math.max(...xs).toFixed(3)}]s · ${xs.length} 点 · 视频fps=${this.videoProcessor.getFPS()}`);
        }

        // 图表视图控制启用（默认全范围视图，聚焦/全览按钮可选切换）
        this.btnViewFirst.disabled = false;
        this.btnViewAll.disabled = false;
        this.btnViewFirst.classList.remove('active');
        this.btnViewAll.classList.remove('active');

        this.updateProgress(100, '完成');
        this.log('success', `图表绘制完成，拟合 R²: ${this.fitRSquared.toFixed(4)}`);
    }

    setChartView(mode) {
        const full = this.chartRender.getFullXMax();
        if (mode === 'all' && full) {
            this.chartRender.setXRange(0, full);
            this.btnViewFirst.classList.remove('active');
            this.btnViewAll.classList.add('active');
        } else if (mode === 'first') {
            // 聚焦：有效振动区间的开头几波（放大查看）
            this.chartRender.focusActiveRange();
            this.btnViewFirst.classList.add('active');
            this.btnViewAll.classList.remove('active');
        } else {
            this.chartRender.resetXRange();
            this.btnViewFirst.classList.remove('active');
            this.btnViewAll.classList.remove('active');
        }
    }

    queryDisplacement() {
        const time = parseFloat(this.queryTime.value);
        if (isNaN(time) || time < 0) {
            this.showError('请输入有效的时间');
            return;
        }

        const displacement = this.nnFitter.predictAtTime(this.processedData.cleanedData, time);
        this.queryResult.textContent = `位移: ${displacement.toFixed(3)} cm`;
    }

    exportReport() {
        const report = {
            videoInfo: {
                frameCount: this.videoProcessor.getFrameCount(),
                fps: this.videoProcessor.getFPS(),
                duration: this.videoProcessor.getDuration()
            },
            calibration: {
                origin: this.coordinateCalib.getOrigin(),
                scale: this.coordinateCalib.getScale()
            },
            vibrationParams: this.vibrationParams,
            fitQuality: {
                rSquared: this.fitRSquared,
                sampleCount: this.processedData.fitData.length,
                vibrationStartTime: this.processedData.vibrationStartTime
            },
            dataSummary: {
                rawDataCount: this.physicalData.length,
                cleanedDataCount: this.processedData.cleanedData.length,
                period: this.processedData.period
            },
            timestamp: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vibration_report_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.log('success', '报告已导出');
    }

    resetApp() {
        this.cancelSignal.cancelled = true;
        this.videoProcessor.cancel();
        if (this.inferencePool) {
            this.inferencePool.cancel();
            this.inferencePool.dispose();
            this.inferencePool = null;
        }

        this.videoProcessor.cleanup();
        this.yoloDetector.dispose();
        this.nnFitter.reset();
        this.vibrationCalc.reset();
        this.dataProcessor.reset();
        this.coordinateCalib.reset();
        this.chartRender.clearCharts();

        this.videoPreview.src = '';
        this.videoPreview.style.display = 'none';
        this.detectionCanvas.style.display = 'none';
        this.detectionCtx.clearRect(0, 0, this.detectionCanvas.width, this.detectionCanvas.height);
        this.videoInfo.textContent = '等待上传...';

        this.frameCount.textContent = '0';
        this.fpsValue.textContent = '0';
        this.equilibriumPos.textContent = '-';
        this.dampingRatio.textContent = '-';
        this.dampingCoeff.textContent = '-';
        this.periodValue.textContent = '-';
        this.stopTime.textContent = '-';
        this.logDecay.textContent = '-';
        this.rSquared.textContent = '-';

        this.queryTime.value = '';
        this.queryResult.textContent = '-';
        this.queryTime.disabled = true;
        this.btnQuery.disabled = true;

        this.btnStart.disabled = true;
        this.btnExport.disabled = true;
        this.btnViewFirst.disabled = true;
        this.btnViewAll.disabled = true;

        this.progressFill.style.width = '0%';
        this.progressText.textContent = '等待开始...';

        this.logContainer.innerHTML = '';

        this.videoFile = null;
        this.detections = [];
        this.physicalData = [];
        this.processedData = {};
        this.vibrationParams = {};
        this.fitRSquared = null;

        this.setStatus('待机', 'idle');
        this.updateStepIndicator(1);
        this.log('info', '系统已重置');
    }

    updateProgress(percent, text) {
        this.progressFill.style.width = `${Math.min(100, Math.max(0, percent))}%`;
        this.progressText.textContent = text;
    }

    updateStepIndicator(step) {
        document.querySelectorAll('.step').forEach((el, index) => {
            const stepNum = index + 1;
            el.classList.remove('active', 'done');
            if (stepNum < step) {
                el.classList.add('done');
            } else if (stepNum === step) {
                el.classList.add('active');
            }
        });
    }

    log(type, message) {
        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        this.logContainer.appendChild(entry);
        this.logContainer.scrollTop = this.logContainer.scrollHeight;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VibrationApp();
});
