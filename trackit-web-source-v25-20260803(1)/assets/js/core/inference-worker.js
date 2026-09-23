/**
 * 推理 Web Worker：独立加载 ORT 会话，处理主线程提交的帧。
 * 消息协议：
 *   { id, type:'infer', modelPath, modelOpts, imageData:{buf,width,height}, mapping }
 *   → { id, ok:true, detections }
 *   → { id, ok:false, error }
 *   { type:'warmup' } → { id, ok:true } （预加载会话）
 */
importScripts(
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js',
    'yolo-common.js'
);

const CLASSES = ['red_pointer', 'green_scale_5cm'];
const CONF_THRESHOLD = 0.25;

let session = null;
let sessionKey = null;

async function ensureSession(modelPath, modelOpts) {
    const key = modelPath + '|' + JSON.stringify(modelOpts);
    if (!session || sessionKey !== key) {
        // Worker 内无 document，ORT 无法自动定位 wasm，必须显式指定（本地文件，离线可用）
        ort.env.wasm.wasmPaths = '/assets/wasm/';
        ort.env.wasm.simd = true;
        // 若页面为 crossOriginIsolated，Worker 内也可用多线程
        if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) {
            ort.env.wasm.numThreads = Math.max(1, (navigator.hardwareConcurrency || 4) / 2);
        }
        session = await ort.InferenceSession.create(modelPath, modelOpts);
        sessionKey = key;
    }
    return session;
}

function infer(imageData, mapping, modelPath, modelOpts) {
    return ensureSession(modelPath, modelOpts).then((sess) => {
        const { buf, width, height } = imageData;
        const pixels = new Uint8ClampedArray(buf);
        const float32 = YoloPost.toNCHW({ data: pixels, width, height });
        const tensor = new ort.Tensor('float32', float32, [1, 3, height, width]);
        const feeds = {};
        feeds[sess.inputNames[0]] = tensor;
        return sess.run(feeds).then((outputs) => {
            const detections = YoloPost.postprocess(
                outputs.output0.data, mapping, CLASSES, CONF_THRESHOLD
            );
            return detections;
        });
    });
}

onmessage = (e) => {
    const msg = e.data;
    if (msg.type === 'warmup') {
        ensureSession(msg.modelPath, msg.modelOpts)
            .then(() => postMessage({ id: msg.id, type: 'warmup-result', ok: true }))
            .catch((err) => postMessage({ id: msg.id, type: 'warmup-result', ok: false, error: err.message }));
        return;
    }
    infer(msg.imageData, msg.mapping, msg.modelPath, msg.modelOpts)
        .then((detections) => postMessage({ id: msg.id, ok: true, detections }))
        .catch((err) => postMessage({ id: msg.id, ok: false, error: err.message }));
};
