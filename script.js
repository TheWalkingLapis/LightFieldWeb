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
}

async function render() {
  await evaluate();

  display_output("rgb");
  display_output("xyz");
}

async function evaluate() {
  const rand_c2w = random_c2w_360();
  log(VB.STATUS, "Random Pose: ", rand_c2w)
  const pts = await sample(rand_c2w); // TODO use pts from gpubuffer if reshape is baked into embedder

  const start = new Date();
  await R2LEngine.run({ input: pts }, { rgb: gpu_tensors["rgb"], xyz: gpu_tensors["xyz"] });
  const end = new Date();

  const inferenceTime = (end.getTime() - start.getTime())/1000;
  log(VB.TIME, "R2L Inference Time: ", inferenceTime);

}

async function sample(c2w) {
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

function random_c2w_360() {
  function normalize(vec) {
    const len = Math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2);
    return vec.map(v => v / len);
  }

  function cross(a, b) {
    return [
      a[1]*b[2] - a[2]*b[1],
      a[2]*b[0] - a[0]*b[2],
      a[0]*b[1] - a[1]*b[0]
    ];
  }

  function randn_bm() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  const radius = 1.5;
  const minTheta = 0.017453292519943295; // ~1 deg
  const maxTheta = 1.5707963267948966;   // 90 deg
  const phiMin = -178 * Math.PI / 180;
  const phiMax = 178 * Math.PI / 180;

  // Sample theta with Gaussian, fallback to uniform if out of range
  const mu = 0.5 * (minTheta + maxTheta);
  const sigma = 0.5 * (maxTheta - minTheta);
  let theta = mu + sigma * randn_bm();  // randn_bm() gives standard normal
  if (theta < minTheta || theta > maxTheta) {
      theta = Math.random() * (maxTheta - minTheta) + minTheta;
  }

  // Sample phi uniformly
  const phi = Math.random() * (phiMax - phiMin) + phiMin;

  // Convert spherical to Cartesian
  const x = radius * Math.sin(theta) * Math.cos(phi);
  const y = radius * Math.sin(theta) * Math.sin(phi);
  const z = radius * Math.cos(theta);
  const center = [x, y, z];

  // Lookat axes
  let forward = normalize(center.map(v => -v)); // look at origin
  let up = [0, 0, -1];
  let right = normalize(cross(up, forward));
  up = normalize(cross(forward, right));

  // Compose 3x4 pose matrix
  const ngp_factor = 4.03112885717555/1.5;
  const pose = [
      [right[0], -up[0], -forward[0], ngp_factor * center[0]],
      [right[1], -up[1], -forward[1], ngp_factor * center[1]],
      [right[2], -up[2], -forward[2], ngp_factor * center[2]]
  ];

  return pose;
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
    canvas.addEventListener("click", (event) => {
      log(VB.STATUS, "Re-Render with new pose");
      render();
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