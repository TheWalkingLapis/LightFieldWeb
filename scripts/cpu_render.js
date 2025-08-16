let cpu_canvas_struct = {};

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
  canvas.id = key + "_cpu";
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
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);

  cpu_canvas_struct[key] = {"ctx": ctx, "imageData": imageData};
}

async function display_output_cpu(key) {

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