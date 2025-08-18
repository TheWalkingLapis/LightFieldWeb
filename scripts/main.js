const canvas_callbacks = true;

const backend = 'webgpu';
let device;

let platform;
let browser;
const supported_browsers = ["Edge", "Chrome"];

let camera;

const VB = {
  ALL: 100,
  STATUS: 50,
  INFO: 40,
  TIME: 20,
  ERROR: 1,
  NONE: 0
}
// everything >= verbose_level is printed
const verbose_level = VB.STATUS;

const RENDER_MODES = {
  CPU: "CPU",
  GPU: "GPU"
}
let render_mode = RENDER_MODES.GPU;

async function start_demo() {
  await init();
  await render();
}

async function init() {
  log(VB.STATUS, "Initalizing ...")

  if (!navigator.gpu) {
    log(VB.ERROR, "WebGPU is not supported on this browser.");
    return;
  }

  platform = get_platform();
  browser = get_browser_name();

  log(VB.INFO, "Detected browser:", browser);
  log(VB.INFO, "Detected platform:", platform);
  
  if (!supported_browsers.includes(browser)) {
    log(VB.ERROR, "Detected Browser '", browser, "' is not supported and might not work properly. Use one of these browser: ", ...supported_browsers);
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

  await create_gpu_canvas("rgb");
  await create_gpu_canvas("xyz");

  const cpu_canvas_div = document.getElementById("cpuCanvasDiv");
  const gpu_canvas_div = document.getElementById("gpuCanvasDiv");
  gpu_canvas_div.appendChild(gpu_canvas_struct["rgb"]["ctx"].canvas);
  gpu_canvas_div.appendChild(gpu_canvas_struct["xyz"]["ctx"].canvas);

  cpu_canvas_div.appendChild(cpu_canvas_struct["rgb"]["ctx"].canvas);
  cpu_canvas_div.appendChild(cpu_canvas_struct["xyz"]["ctx"].canvas);

  log(VB.STATUS, "Finished cpu canvas creation.")

  const toggleButton = document.getElementById("toggleRenderer");

  toggleButton.addEventListener("click", () => {
    render_mode = (render_mode == RENDER_MODES.CPU) ? RENDER_MODES.GPU : RENDER_MODES.CPU; // toggle the variable

    log(VB.INFO, "Renderer switched to",  (render_mode == RENDER_MODES.CPU) ? "CPU" : "GPU");
  });

  camera = new Camera(0.0, 0.0);
}

async function render() {
  await evaluate();

  switch (render_mode) {
    case RENDER_MODES.CPU:
      const display_start_cpu = new Date();
      await Promise.all([
        display_output_cpu("rgb"),
        display_output_cpu("xyz")
      ]);
      const display_end_cpu = new Date();
      const displayTimeCPU = (display_end_cpu.getTime() - display_start_cpu.getTime())/1000;
      log(VB.TIME, "GPU->CPU Time: ", displayTimeCPU);
      break;
    case RENDER_MODES.GPU:
      const display_start_gpu = new Date();
      await Promise.all([
        display_output_gpu("rgb"),
        display_output_gpu("xyz")
      ]);
      const display_end_gpu = new Date();
      const displayTimeGPU = (display_end_gpu.getTime() - display_start_gpu.getTime())/1000;
      log(VB.TIME, "Render Time (GPU): ", displayTimeGPU);
  }
}