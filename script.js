let Sampler;
let Embedder;
let R2LEngine;

let pts_tensor;
let embb_pts_tensor;
let gpu_tensors = {};
let gpu_buffers = {};

let cpu_canvas_struct = {};
const render_on_click = true;

const backend = 'webgpu';
let device;

let camera;

const VB = {
  NONE: 100,
  LOG: 90,
  TIME: 80,
  STATUS: 20,
  ALL: 0
}
// everything >= verbose_level is printed
const verbose_level = VB.ALL;

async function start_demo() {
  await init();
  await render();
}

async function init() {
  log(VB.STATUS, "Initalizing ...")

  if (!navigator.gpu) {
    log(VB.ALL, "WebGPU is not supported on this browser.");
    return;
  }

  Sampler = await ort.InferenceSession.create('./models/opset_11/Sampler.onnx', {
    executionProviders: [backend]
  });
  Embedder = await ort.InferenceSession.create('./models/opset_11/Embedder.onnx', {
    executionProviders: [backend]
  });
  R2LEngine = await ort.InferenceSession.create('./models/opset_11/ckpt.onnx', {
    executionProviders: [backend]
  });
  
  // device is definetly ready after session creation
  device = ort.env.webgpu.device;

  log(VB.STATUS, "Finished inference session creation.")

  sessions = [Sampler, Embedder, R2LEngine];
  strings = ["Sampler", "Embedder", "R2LEngine"];
  // List all input/output names and shapes
  sessions.forEach(session => {
    log(VB.STATUS, strings[sessions.indexOf(session)], ":")
    session.inputNames.forEach(name => {
        const inputMeta = session.inputMetadata[0];
        log(VB.STATUS, `Input name: ${name}, shape:`, inputMeta.shape, "type:", inputMeta.type);
    });
    session.outputNames.forEach(name => {
      const outputMeta = session.outputMetadata[0];
      log(VB.STATUS, `Output name: ${name}, shape:`, outputMeta.shape, "type:", outputMeta.type);
    });
  });

  pts_tensor = gpu_tensor_from_dims("pts", [10000, 24]);
  embb_pts_tensor = gpu_tensor_from_dims("embb_pts", [24, 13, 10000]);
  rgb_tensor = gpu_tensor_from_dims("rgb", [1, 3, 800, 800]);
  xyz_tensor = gpu_tensor_from_dims("xyz", [1, 3, 800, 800]);

  log(VB.STATUS, "Finished gpu tensor creation.")

  await create_cpu_canvas("rgb");
  await create_cpu_canvas("xyz");

  document.body.appendChild(cpu_canvas_struct["rgb"]["ctx"].canvas);
  document.body.appendChild(cpu_canvas_struct["xyz"]["ctx"].canvas);

  log(VB.STATUS, "Finished cpu canvas creation.")

  camera = new Camera(0.0, 0.0);
}

async function render() {
  await evaluate();

  display_output("rgb");
  display_output("xyz");
}

async function evaluate() {
  const pts = await sample(); // TODO use pts from gpubuffer if reshape is baked into embedder

  const start = new Date();
  await R2LEngine.run({ input: pts }, { rgb: gpu_tensors["rgb"], xyz: gpu_tensors["xyz"] });
  const end = new Date();

  const inferenceTime = (end.getTime() - start.getTime())/1000;
  log(VB.TIME, "R2L Inference Time: ", inferenceTime);

}

async function sample() {
  const c2w = camera.get_c2w_as_input();
  const c2w33 = new ort.Tensor('float32', c2w.map(row => row.slice(0, 3)).flat(), [3, 3]);
  const c2w13 = new ort.Tensor('float32', c2w.map(row => [row[3]]).flat(), [1, 3]);

  const sample_start = new Date();
  await Sampler.run({ origin: c2w33, direction: c2w13}, { pts: gpu_tensors["pts"] });
  const sample_end = new Date();

  const embb_start = new Date();
  await Embedder.run({ pts: gpu_tensors["pts"] }, { embbpts: gpu_tensors["embb_pts"] });
  const embb_end = new Date();

  const sampleInferenceTime = (sample_end.getTime() - sample_start.getTime())/1000;
  const embbInferenceTime = (embb_end.getTime() - embb_start.getTime())/1000;
  log(VB.TIME, "Sample Time: ", sampleInferenceTime);
  log(VB.TIME, "Embedding Time: ", embbInferenceTime);

  return gpu_tensors["embb_pts"].reshape([1, 312, 100, 100]);
}

function gpu_tensor_from_dims(key, dims) {
  let n = 4; // float = 4 byte
  dims.forEach(i => n *= i);

  const buffer = device.createBuffer({
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    size: Math.ceil(n / 16) * 16 /* align to 16 bytes */
  });
  const tensor = ort.Tensor.fromGpuBuffer(buffer, {
    dataType: 'float32',
    dims: dims
  });
  
  gpu_buffers[key] = buffer;
  gpu_tensors[key] = tensor;

  return tensor;
}

async function gpu_to_cpu(key) {
  const buffer = gpu_buffers[key];

  // Create CPU-readable buffer
  const readBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });

  // Copy data from GPU-only buffer to CPU-readable buffer
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
  device.queue.submit([encoder.finish()]);

  // Map and read
  await readBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = readBuffer.getMappedRange();
  const cpuData = new Float32Array(copyArrayBuffer);

  return [cpuData, readBuffer];
}

async function create_cpu_canvas(key) {
  const [channels, height, width] = [3, 800, 800];

  const canvas = document.createElement("canvas");
  canvas.id = key;
  canvas.width = width;
  canvas.height = height;
  if (render_on_click) {
    canvas.addEventListener("mousedown", (event) => {
      camera.mousedown_hook(event);
    });
    canvas.addEventListener("mouseup", (event) => {
      camera.mouseup_hook(event);
    });
    canvas.addEventListener("mousemove", (event) => {
      camera.mousemove_hook(event, render);
    });
  }
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);

  cpu_canvas_struct[key] = {"ctx": ctx, "imageData": imageData};
}

async function display_output(key) {

  const [data, unmap_buffer] = await gpu_to_cpu(key);
  const [channels, height, width] = [3, 800, 800];
  const ctx = cpu_canvas_struct[key]["ctx"];
  const imageData = cpu_canvas_struct[key]["imageData"];

  const pixels = imageData.data;

  let pixelIndex = 0;
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      // ONNX CHW format: R=channel0, G=channel1, B=channel2
      const r = data[0 * height * width + h * width + w];
      const g = data[1 * height * width + h * width + w];
      const b = data[2 * height * width + h * width + w];

      pixels[pixelIndex++] = Math.min(255, Math.max(0, r * 255)); // R
      pixels[pixelIndex++] = Math.min(255, Math.max(0, g * 255)); // G
      pixels[pixelIndex++] = Math.min(255, Math.max(0, b * 255)); // B
      pixels[pixelIndex++] = 255; // Alpha
    }
  }

  ctx.putImageData(imageData, 0, 0);

  unmap_buffer.unmap();
}

function compareTensors(t1, t2) {
  if (t1.length !== t2.length) {
    throw new Error(`Tensor size mismatch: ${t1.length} vs ${t2.length}`);
  }
  let epsilon = 0.00001;
  let maxDiff = 0;
  let sumSq = 0;
  for (let i = 0; i < t1.length; i++) {
    const diff = Math.abs(t1[i] - t2[i]);
    if (diff > epsilon) {
      if (diff > maxDiff) maxDiff = diff;
      sumSq += diff * diff;
    }
  }
  const mse = sumSq / t1.length;
  return [maxDiff, mse];
}

function log(verbosity, ...txt) {
  if (verbosity >= verbose_level) {
    console.log(...txt)
  }
}