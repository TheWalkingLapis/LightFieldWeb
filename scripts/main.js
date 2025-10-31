const canvas_callbacks = true;

const backend = 'webgpu';
let device;

let platform;
let browser;
const supported_browsers = ["Edge", "Chrome"];

let camera;
let light_source;

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
  LIGHTING: "LIGHTING"
}
let render_mode = RENDER_MODES.GPU;

let times = {"eval": [], "render": []};

async function start_demo() {
  await init();
  await render(1);
  // reset timers after inital render
  times = {"eval": [], "render": []};
  for (let idx = 0; idx < 25; idx++) {
    await render(idx);
  }
  console.log(times);
  Object.entries(times).forEach(timer => {
    let key = timer[0];
    let time_arr = timer[1];
    console.log(time_arr);
    let time_acc = 0;
    time_arr.forEach(time => {
      time_acc += time;
    })
    let mean_time = time_acc / time_arr.length;
    console.log(key, ": ", mean_time);
  });
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

  Sampler = await ort.InferenceSession.create('./models/MR2L/Sampler.onnx', {
    executionProviders: [backend]
  });
  Embedder = await ort.InferenceSession.create('./models/MR2L/Embedder.onnx', {
    executionProviders: [backend]
  });
  R2LEngine = await ort.InferenceSession.create('./models/MR2L/ckpt.onnx', {
    executionProviders: [backend]
  });
  
  // device is definetly ready after session creation
  device = ort.env.webgpu.device;
  //ort.env.trace = true;

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

  log(VB.STATUS, "Finished gpu tensor creation.")

  
  camera = new Camera(0.0, 0.0);
  light_source = new Camera(0.0, 0.0);


  await init_cpu_render();

  await init_gpu_render();

  log(VB.STATUS, "Finished canvas creation.")

  let gpuSection = document.getElementById("gpuSection");
  let cpuSection = document.getElementById("cpuSection");

  gpuSection.style.display = "flex";
  cpuSection.style.display = "none";

  let toggleButton = document.getElementById("toggleCPURenderer");
  toggleButton.addEventListener("click", () => {
    render_mode = RENDER_MODES.CPU;
    gpuSection.style.display = "none";
    cpuSection.style.display = "block";

    log(VB.INFO, "Renderer switched to CPU");
  });
  toggleButton = document.getElementById("toggleLIGHTINGRenderer");
  toggleButton.addEventListener("click", () => {
    render_mode = RENDER_MODES.GPU;
    gpuSection.style.display = "none";
    cpuSection.style.display = "block";

    log(VB.INFO, "Renderer switched to Shader");
  });
}

async function render(index) {
  let path = "pt/gen_images/" + index.toString().padStart(3, '0') + "_c2w.json";
  await evaluate(path);

  switch (render_mode) {
    case RENDER_MODES.CPU:
      const display_start_cpu = performance.now();
      await Promise.all([
        display_output_cpu("rgb")
      ]);
      const display_end_cpu = performance.now();
      const displayTimeCPU = (display_end_cpu - display_start_cpu)/1000;
      log(VB.TIME, "GPU->CPU Time: ", displayTimeCPU);
      break;

    case RENDER_MODES.GPU:
      const display_start_gpu = performance.now();
      await Promise.all([
        display_output_gpu("rgb")
      ]);
      const display_end_gpu = performance.now();
      const displayTimeGPU = (display_end_gpu - display_start_gpu)/1000;
      log(VB.TIME, "Render Time (GPU): ", displayTimeGPU);
      times["render"].push(displayTimeGPU);
      break;

    case RENDER_MODES.LIGHTING:
      const display_start_lighting = performance.now();
      await Promise.all([
        display_output_gpu("rgb")
      ]);
      const display_end_lighting = performance.now();
      const displayTimeLIGHTING = (display_end_lighting - display_start_lighting)/1000;
      log(VB.TIME, "Render Time (GPU): ", displayTimeLIGHTING);
      break;
  }
}