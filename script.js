let Sampler;
let Embedder;
let R2LEngine;

let pts_tensor;
let embb_pts_tensor;
let rgb_tensor;
let xyz_tensor;

const backend = 'webgpu';
let device;

async function init() {
  console.log("Initalizing ...")

  if (!navigator.gpu) {
    console.error("WebGPU is not supported on this browser.");
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

  console.log("Finished inference session creation.")

  sessions = [Sampler, Embedder, R2LEngine];
  strings = ["Sampler", "Embedder", "R2LEngine"];
  // List all input/output names and shapes
  sessions.forEach(session => {
    console.log(strings[sessions.indexOf(session)], ":")
    session.inputNames.forEach(name => {
        const inputMeta = session.inputMetadata[0];
        console.log(`Input name: ${name}, shape:`, inputMeta.shape, "type:", inputMeta.type);
    });
    session.outputNames.forEach(name => {
      const outputMeta = session.outputMetadata[0];
      console.log(`Output name: ${name}, shape:`, outputMeta.shape, "type:", outputMeta.type);
    });
  });

  pts_tensor = gpu_tensor_from_dims([10000, 24]);
  embb_pts_tensor = gpu_tensor_from_dims([24, 13, 10000]);
  rgb_tensor = gpu_tensor_from_dims([1, 3, 800, 800]);
  xyz_tensor = gpu_tensor_from_dims([1, 3, 800, 800]);

  console.log("Finished gpu tensor creation.")

  evaluate();

}

async function evaluate() {
  const rand_c2w = random_c2w_360();
  console.log("Random Pose: ", rand_c2w)
  const pts = await sample(rand_c2w);

  const start = new Date();
  const out = await R2LEngine.run({ input: pts }); // TODO output into buffers
  const end = new Date();
  console.log("RGB dim", out.rgb.dims)
  console.log("XYZ dim", out.xyz.dims)


  display_output(out.rgb.data);
  display_output(out.xyz.data);


  const inferenceTime = (end.getTime() - start.getTime())/1000;
  console.log("R2L Inference Time: ", inferenceTime);

}

async function sample(c2w) {
  const c2w33 = new ort.Tensor('float32', c2w.map(row => row.slice(0, 3)).flat(), [3, 3]);
  const c2w13 = new ort.Tensor('float32', c2w.map(row => [row[3]]).flat(), [1, 3]);

  const sample_start = new Date();
  await Sampler.run({ origin: c2w33, direction: c2w13}, { pts: pts_tensor });
  const sample_end = new Date();
  console.log("Sampler pts dim", pts_tensor.dims)

  const embb_start = new Date();
  await Embedder.run({ pts: pts_tensor }, { embbpts: embb_pts_tensor });
  const embb_end = new Date();

  /*console.log("Embedder pts dim", embbpts.dims)
  const embbpts_in = new ort.Tensor('float32', embbpts.data , [1, 312, 100, 100]);
  console.log("Embedder reshaped pts dim", embbpts_in.dims)*/

  const sampleInferenceTime = (sample_end.getTime() - sample_start.getTime())/1000;
  const embbInferenceTime = (embb_end.getTime() - embb_start.getTime())/1000;
  console.log("Sample Time: ", sampleInferenceTime);
  console.log("Embedding Time: ", embbInferenceTime);

  return embb_pts_tensor.reshape([1, 312, 100, 100]);
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

function gpu_tensor_from_dims(dims) {
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

  return tensor;
}

function display_output(data) {
  const [batch, channels, height, width] = [1, 3, 800, 800];

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);
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
  document.body.appendChild(canvas);
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

init();