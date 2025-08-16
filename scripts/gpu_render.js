let gpu_canvas_struct = {};

async function create_gpu_canvas(key) {
  const [channels, height, width] = [3, 800, 800];

  const canvas = document.createElement("canvas");
  canvas.id = key + "gpu";
  canvas.width = width;
  canvas.height = height;
  if (canvas_callbacks) {
    if (platform == "Mobile") {
      canvas.addEventListener("touchstart", (event) => {
        event.preventDefault();
        const touch = event.touches[0];
        camera.mousedown_hook({ clientX: touch.clientX, clientY: touch.clientY });
      });
      canvas.addEventListener("touchend", (event) => {
        event.preventDefault();
        const touch = event.touches[0];
        camera.mouseup_hook({});
      });
      canvas.addEventListener("touchmove", (event) => {
        event.preventDefault();
        const touch = event.touches[0];
        camera.mousemove_hook({ clientX: touch.clientX, clientY: touch.clientY }, render);
      });
    } else {
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
  }
  const ctx = canvas.getContext("webgpu");
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  gpu_canvas_struct[key] = {"ctx": ctx};
}

async function display_output_gpu(key) {
  log(VB.INFO, "GPU Rendering not implemented yet, TODO")
}