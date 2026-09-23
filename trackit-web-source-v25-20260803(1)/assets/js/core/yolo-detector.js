class YoloDetector {
    constructor() {
        this.session = null;
        this.modelPath = 'models/yolov8-tiny.onnx';
        // 本项目模型导出为固定 1280×1280 输入，故默认直接使用模型尺寸。
        // 仅当模型元数据显示输入为动态尺寸时，才启用 640 快速路径 + 1280 复查。
        this.inputSize = 1280;
        this.maxInputSize = 1280;
        this.adaptive = false;
        this.confidenceThreshold = 0.25;
        // 640 结果中指针置信度达到该值即接受，否则用 1280 重检该帧
        this.fastAcceptConfidence = 0.5;
        this.classes = ['red_pointer', 'green_scale_5cm'];
        this.isLoaded = false;
        this._floatBuffer = null;
    }

    async checkModelFile() {
        try {
            const response = await fetch(this.modelPath, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error(`模型文件不可访问: ${response.status}`);
            }
            return true;
        } catch (error) {
            throw new Error(`无法访问模型文件: ${this.modelPath}`);
        }
    }

    async loadModel() {
        if (this.isLoaded && this.session) {
            return true;
        }

        try {
            await this.checkModelFile();

            // 多线程需要 COOP/COEP 响应头提供 SharedArrayBuffer，否则保持单线程
            if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
                ort.env.wasm.numThreads = Math.max(1, navigator.hardwareConcurrency || 4);
            }
            ort.env.wasm.simd = true;
            ort.env.wasm.wasmPaths = '/assets/wasm/';   // 本地 wasm，离线可用

            const options = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all',
                enableCpuMemArena: true,
                logSeverityLevel: 3
            };
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('模型加载超时')), 60000);
            });

            const loadPromise = ort.InferenceSession.create(this.modelPath, options);
            this.session = await Promise.race([loadPromise, timeoutPromise]);

            this.isLoaded = true;
            console.log('YOLO模型加载成功');

            if (this.session.inputNames && this.session.inputNames.length > 0) {
                this.inputName = this.session.inputNames[0];
                this._resolveInputSize();
            }
            return true;
        } catch (error) {
            console.error('YOLO模型加载失败:', error);
            let errorMsg = `模型加载失败: ${error.message}`;
            if (error.message.includes('Failed to fetch')) {
                errorMsg += '\n\n可能的原因：\n' +
                    '1. 请确保使用HTTP服务器运行（如localhost），而非直接打开HTML文件\n' +
                    '2. 检查模型文件路径是否正确: models/yolov8-tiny.onnx\n' +
                    '3. 确保模型文件存在且可访问';
            } else if (error.message.includes('timeout')) {
                errorMsg += '\n\n可能的原因：网络较慢，请尝试刷新页面';
            }
            throw new Error(errorMsg);
        }
    }

    /**
     * 根据模型输入元数据决定输入尺寸（可选）：
     * 元数据可用且为动态 H/W → 启用 640 快速路径 + 1280 复查；
     * 元数据可用且固定 H/W → 使用模型尺寸；
     * 元数据缺失 → 保持默认 1280（本项目模型固定 1280，喂其他尺寸会报维度错误）。
     */
    _resolveInputSize() {
        const meta = this.session.inputMetadata?.[this.inputName];
        const dims = meta?.dims;
        if (Array.isArray(dims) && dims.length >= 4) {
            if (dims[2] > 0 && dims[3] > 0) {
                this.inputSize = dims[2];
                this.maxInputSize = dims[3];
                this.adaptive = false;
                console.log(`模型固定输入尺寸: ${dims[2]}×${dims[3]}`);
            } else {
                this.inputSize = 640;
                this.maxInputSize = 1280;
                this.adaptive = true;
                console.log('模型输入为动态尺寸，启用 640/1280 自适应');
            }
        }
    }

    /** 当前应使用的抓帧目标尺寸（视频处理器调用） */
    get captureTargetSize() {
        return this.inputSize;
    }

    /**
     * 推理单帧。imageData 必须已 letterbox 到 inputSize×inputSize（由 VideoProcessor 绘制）。
     * mapping: { scale, padLeft, padTop } 将模型坐标映射回视频像素坐标。
     */
    async detect(imageData, mapping) {
        if (!this.session) {
            throw new Error('模型未加载');
        }

        const inputTensor = this.preprocess(imageData);
        try {
            const feeds = {};
            feeds[this.inputName || 'images'] = inputTensor;
            const outputs = await this.session.run(feeds);
            const results = this.postprocess(
                outputs.output0.data,
                mapping || { scale: 1, padLeft: 0, padTop: 0 }
            );
            return results;
        } catch (error) {
            console.error(`[DEBUG] 推理失败: ${error.message}`);
            throw error;
        } finally {
            inputTensor.dispose();
        }
    }

    /**
     * 像素转换：ImageData(data) → NCHW Float32Array，复用缓冲避免每次分配
     */
    preprocess(imageData) {
        const { width, height } = imageData;
        this._floatBuffer = YoloPost.toNCHW(imageData, this._floatBuffer);
        return new ort.Tensor('float32', this._floatBuffer, [1, 3, height, width]);
    }

    /** 共享后处理（与推理 Worker 同一份实现） */
    postprocess(output, mapping) {
        return YoloPost.postprocess(output, mapping, this.classes, this.confidenceThreshold);
    }

    drawDetections(ctx, detections, showLabels = true) {
        detections.forEach(detection => {
            const isPointer = detection.class === 'red_pointer';
            ctx.strokeStyle = isPointer ? '#ff4d4f' : '#52c41a';
            ctx.lineWidth = 2;
            ctx.strokeRect(detection.x1, detection.y1, detection.width, detection.height);

            ctx.fillStyle = isPointer ? '#ff4d4f' : '#52c41a';
            ctx.beginPath();
            ctx.arc(detection.x, detection.y, 4, 0, Math.PI * 2);
            ctx.fill();

            if (showLabels) {
                const label = `${detection.class === 'red_pointer' ? '指针' : '标尺'}: ${(detection.confidence * 100).toFixed(0)}%`;
                ctx.font = '12px Consolas, monospace';
                const textWidth = ctx.measureText(label).width;
                ctx.fillStyle = isPointer ? 'rgba(255, 77, 79, 0.9)' : 'rgba(82, 196, 26, 0.9)';
                ctx.fillRect(detection.x1, detection.y1 - 18, textWidth + 8, 16);
                ctx.fillStyle = '#ffffff';
                ctx.fillText(label, detection.x1 + 4, detection.y1 - 6);
            }
        });
    }

    dispose() {
        if (this.session) {
            this.session.dispose();
            this.session = null;
        }
        this.isLoaded = false;
        this._floatBuffer = null;
    }
}
