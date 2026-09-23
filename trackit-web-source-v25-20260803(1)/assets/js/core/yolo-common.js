/**
 * YOLO 后处理共享模块：主线程与推理 Worker 共用同一份实现。
 * 定义全局 YoloPost = { postprocess, nms, calculateIoU }
 */
(function (global) {
    'use strict';

    /** 将模型输出转为检测框数组（模型坐标 → 视频像素坐标由 mapping 换算） */
    function postprocess(output, mapping, classes, confidenceThreshold) {
        const { scale = 1, padLeft = 0, padTop = 0, videoWidth = Infinity, videoHeight = Infinity } = mapping || {};
        const boxes = [];
        const numBoxes = output.length / 6;

        for (let i = 0; i < numBoxes; i++) {
            const base = i * 6;
            const xCenter = output[base] / scale - padLeft;
            const yCenter = output[base + 1] / scale - padTop;
            const w = output[base + 2] / scale;
            const h = output[base + 3] / scale;
            const confidence = output[base + 4];
            const classId = Math.round(output[base + 5]);

            if (confidence >= confidenceThreshold && classId < classes.length) {
                boxes.push({
                    class: classes[classId],
                    classId,
                    confidence,
                    x: xCenter,
                    y: yCenter,
                    x1: Math.max(0, xCenter - w / 2),
                    y1: Math.max(0, yCenter - h / 2),
                    x2: Math.min(videoWidth, xCenter + w / 2),
                    y2: Math.min(videoHeight, yCenter + h / 2),
                    width: w,
                    height: h
                });
            }
        }

        return nms(boxes, 0.45);
    }

    function nms(boxes, iouThreshold) {
        if (boxes.length === 0) return [];

        boxes.sort((a, b) => b.confidence - a.confidence);

        const selected = [];
        const suppressed = new Set();

        for (let i = 0; i < boxes.length; i++) {
            if (suppressed.has(i)) continue;
            selected.push(boxes[i]);

            for (let j = i + 1; j < boxes.length; j++) {
                if (suppressed.has(j) || boxes[i].classId !== boxes[j].classId) continue;
                if (calculateIoU(boxes[i], boxes[j]) > iouThreshold) {
                    suppressed.add(j);
                }
            }
        }

        return selected;
    }

    function calculateIoU(box1, box2) {
        const x1 = Math.max(box1.x1, box2.x1);
        const y1 = Math.max(box1.y1, box2.y1);
        const x2 = Math.min(box1.x2, box2.x2);
        const y2 = Math.min(box1.y2, box2.y2);

        const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        const area1 = box1.width * box1.height;
        const area2 = box2.width * box2.height;
        const union = area1 + area2 - intersection;

        return union > 0 ? intersection / union : 0;
    }

    /** ImageData 像素 → NCHW Float32Array（复用到传入的缓冲） */
    function toNCHW(imageData, buffer) {
        const { data, width, height } = imageData;
        const n = width * height;
        if (!buffer || buffer.length !== n * 3) {
            buffer = new Float32Array(n * 3);
        }
        const inv = 1 / 255;
        for (let i = 0, j = 0; i < data.length; i += 4, j++) {
            buffer[j] = data[i] * inv;
            buffer[n + j] = data[i + 1] * inv;
            buffer[2 * n + j] = data[i + 2] * inv;
        }
        return buffer;
    }

    global.YoloPost = { postprocess, nms, calculateIoU, toNCHW };
})(typeof self !== 'undefined' ? self : this);
