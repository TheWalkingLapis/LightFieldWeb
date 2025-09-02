let cpu_canvas_struct = {};

async function init_cpu_render() {
    await create_cpu_canvas("rgb");
    await create_cpu_canvas("xyz");
    await create_cpu_canvas("normal");
    await create_cpu_canvas("mask");

      
    const cpu_canvas_div = document.getElementById("cpuCanvasDiv");

    cpu_canvas_div.appendChild(cpu_canvas_struct["rgb"]["ctx"].canvas);
    cpu_canvas_div.appendChild(cpu_canvas_struct["xyz"]["ctx"].canvas);
    cpu_canvas_div.appendChild(cpu_canvas_struct["normal"]["ctx"].canvas);
    cpu_canvas_div.appendChild(cpu_canvas_struct["mask"]["ctx"].canvas);
}

async function gpu_to_cpu(key) {
  const buffer = gpu_tensors[key].gpuBufferData;

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
  canvas.id = key + "_cpu";
  canvas.width = width;
  canvas.height = height;
  if (canvas_callbacks) {
    connect_to_canvas(camera, light_source, canvas);
  }
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);

  cpu_canvas_struct[key] = {"ctx": ctx, "imageData": imageData};
}

async function display_output_cpu(key) {

  const [data, unmap_buffer] = await gpu_to_cpu(key);
  const [channels, height, width] = [key == "mask" ? 1 : 3, 800, 800];
  const ctx = cpu_canvas_struct[key]["ctx"];
  const imageData = cpu_canvas_struct[key]["imageData"];

  const pixels = imageData.data;

  let pixelIndex = 0;
  for (let h = 0; h < height; h++) {
    for (let w = 0; w < width; w++) {
      const idx = (h * width + w) * channels
      const r = data[idx + 0];
      const g = data[idx + 1];
      const b = data[idx + 2];

      pixels[pixelIndex++] = Math.min(255, Math.max(0, r * 255));
      pixels[pixelIndex++] = Math.min(255, Math.max(0, g * 255));
      pixels[pixelIndex++] = Math.min(255, Math.max(0, b * 255));
      pixels[pixelIndex++] = 255;
    }
  }

  ctx.putImageData(imageData, 0, 0);

  unmap_buffer.unmap();
}

async function connect_to_canvas(camera, light, canvas) {
    if (platform == "Mobile") {
        canvas.addEventListener("touchstart", (event) => {
          event.preventDefault();
          const touch = event.touches[0];
          if (event.ctrlKey) {
            light.mousedown_hook({ clientX: touch.clientX, clientY: touch.clientY });
          } else {
            camera.mousedown_hook({ clientX: touch.clientX, clientY: touch.clientY });
          }
        });
        canvas.addEventListener("touchend", (event) => {
          event.preventDefault();
          const touch = event.touches[0];
          if (event.ctrlKey) {
            light.mouseup_hook({});
          } else {
            camera.mouseup_hook({});
          }
        });
        canvas.addEventListener("touchmove", (event) => {
          event.preventDefault();
          const touch = event.touches[0];
          if (event.ctrlKey) {
            light.mousemove_hook({ clientX: touch.clientX, clientY: touch.clientY }, render);
          } else {
            camera.mousemove_hook({ clientX: touch.clientX, clientY: touch.clientY }, render);
          }
        });
      } else {
        canvas.addEventListener("mousedown", (event) => {
          if (event.ctrlKey) {
            light.mousedown_hook(event);
          } else{
            camera.mousedown_hook(event);
          }
        });
        canvas.addEventListener("mouseup", (event) => {
          if (event.ctrlKey) {
            light.mouseup_hook(event);
          } else{
            camera.mouseup_hook(event);
          }
        });
        canvas.addEventListener("mousemove", (event) => {
          if (event.ctrlKey) {
            light.mousemove_hook(event, render);
          } else{
            camera.mousemove_hook(event, render);
          }
        });
      }
  }