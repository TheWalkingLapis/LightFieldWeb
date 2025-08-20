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
  GPU: "GPU",
  //LIGHTING: "LIGHTING"
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
  R2LEngine = await ort.InferenceSession.create('./models/opset_11/reshape_output/ckpt.onnx', {
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
  rgb_tensor = gpu_tensor_from_dims("rgb", [1, 800, 800, 3]);
  xyz_tensor = gpu_tensor_from_dims("xyz", [1, 800, 800, 3]);

  log(VB.STATUS, "Finished gpu tensor creation.")

  await create_cpu_canvas("rgb");
  await create_cpu_canvas("xyz");

  await init_gpu_render();

  const cpu_canvas_div = document.getElementById("cpuCanvasDiv");

  cpu_canvas_div.appendChild(cpu_canvas_struct["rgb"]["ctx"].canvas);
  cpu_canvas_div.appendChild(cpu_canvas_struct["xyz"]["ctx"].canvas);

  log(VB.STATUS, "Finished canvas creation.")

  let toggleButton = document.getElementById("toggleCPURenderer");
  toggleButton.addEventListener("click", () => {
    render_mode = RENDER_MODES.CPU;

    log(VB.INFO, "Renderer switched to CPU");
  });
  toggleButton = document.getElementById("toggleGPURenderer");
  toggleButton.addEventListener("click", () => {
    render_mode = RENDER_MODES.GPU;

    log(VB.INFO, "Renderer switched to GPU");
  });
  toggleButton = document.getElementById("toggleLIGHTINGRenderer");
  toggleButton.addEventListener("click", () => {
    render_mode = RENDER_MODES.LIGHTING;

    log(VB.INFO, "Renderer switched to Shader");
  });

  camera = new Camera(0.0, 0.0);
}

async function render() {
  await evaluate();

  switch (render_mode) {
    case RENDER_MODES.CPU:
      const display_start_cpu = performance.now();
      await Promise.all([
        display_output_cpu("rgb"),
        display_output_cpu("xyz")
      ]);
      const display_end_cpu = performance.now();
      const displayTimeCPU = (display_end_cpu - display_start_cpu)/1000;
      log(VB.TIME, "GPU->CPU Time: ", displayTimeCPU);
      break;
    case RENDER_MODES.GPU:
      const display_start_gpu = performance.now();
      await Promise.all([
        display_output_gpu("rgb"),
        display_output_gpu("xyz")
      ]);
      const display_end_gpu = performance.now();
      const displayTimeGPU = (display_end_gpu - display_start_gpu)/1000;
      log(VB.TIME, "Render Time (GPU): ", displayTimeGPU);
      break;
    case RENDER_MODES.LIGHTING:
      const display_start_lighting = performance.now();
      await Promise.all([
        display_output_gpu("lighting")
      ]);
      const display_end_lighting = performance.now();
      const displayTimeLIGHTING = (display_end_lighting - display_start_lighting)/1000;
      log(VB.TIME, "Render Time (GPU): ", displayTimeLIGHTING);
      break;
  }
}