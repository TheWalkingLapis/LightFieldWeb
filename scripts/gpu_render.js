let gpu_canvas_struct = {};

let buffer_to_texture_shader_code;
let render_texture_shader_code;
let lighting_shader_code;

async function create_gpu_canvas(key) {
  const [channels, height, width] = [3, 800, 800];

  const canvas = document.createElement("canvas");
  canvas.id = key + "_gpu";
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
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  gpu_canvas_struct[key] = {"ctx": ctx};
}

async function display_output_gpu(key = "") {
  
  const format = navigator.gpu.getPreferredCanvasFormat();

  switch (render_mode) {
    case RENDER_MODES.GPU:
      const context = gpu_canvas_struct[key]["ctx"];

      const display_texture = device.createTexture({
        size: [800, 800],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
      });

      await dispatch_buffer_to_texture(key, display_texture);

      const render_shaderModule = device.createShaderModule({ code: render_texture_shader_code });

      const render_pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: render_shaderModule, entryPoint: "vsMain" },
        fragment: {
          module: render_shaderModule,
          entryPoint: "fsMain",
          targets: [{ format }]
        },
        primitive: { topology: "triangle-list" }
      });

      const render_sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
      });

      const render_bindGroup = device.createBindGroup({
        layout: render_pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: display_texture.createView() },
          { binding: 1, resource: render_sampler }
        ]
      });

      const render_encoder = device.createCommandEncoder();

      const render_pass = render_encoder.beginRenderPass({
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store"
        }]
      });

      render_pass.setPipeline(render_pipeline);
      render_pass.setBindGroup(0, render_bindGroup);
      render_pass.draw(6);
      render_pass.end();

      const render_start = new Date();
      device.queue.submit([render_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const render_end = new Date();

      
      const renderTimeGPU = (render_end.getTime() - render_start.getTime())/1000;
      log(VB.TIME, "Render Canvas Time (GPU): ", renderTimeGPU);
      break;


    case RENDER_MODES.LIGHTING:
      const lighting_context = gpu_canvas_struct[key]["ctx"];

      const rgb_texture = device.createTexture({
        size: [800, 800],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
      });
      const xyz_texture = device.createTexture({
        size: [800, 800],
        format: "rgba8unorm",
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
      });

      await dispatch_buffer_to_texture("rgb", rgb_texture);
      await dispatch_buffer_to_texture("xyz", xyz_texture);

      const lighting_shaderModule = device.createShaderModule({ code: lighting_shader_code });

      const lighting_pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: { module: lighting_shaderModule, entryPoint: "vsMain" },
        fragment: {
          module: lighting_shaderModule,
          entryPoint: "fsMain",
          targets: [{ format }]
        },
        primitive: { topology: "triangle-list" }
      });

      const lighting_sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
      });

      const lighting_uniform_data = new Float32Array([0.0, 10.0, 0.0, 0.0]); // vec3 + padding

      const lighting_uniform_buffer = device.createBuffer({
        size: lighting_uniform_data.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      // Upload the data
      device.queue.writeBuffer(lighting_uniform_buffer, 0, lighting_uniform_data);

      const lighting_bindGroup = device.createBindGroup({
        layout: lighting_pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: rgb_texture.createView() },
          { binding: 1, resource: xyz_texture.createView() },
          { binding: 2, resource: lighting_sampler },
          { binding: 3, resource: { buffer: lighting_uniform_buffer} }
        ]
      });

      const lighting_encoder = device.createCommandEncoder();

      const lighting_pass = lighting_encoder.beginRenderPass({
        colorAttachments: [{
          view: lighting_context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store"
        }]
      });

      lighting_pass.setPipeline(lighting_pipeline);
      lighting_pass.setBindGroup(0, lighting_bindGroup);
      lighting_pass.draw(6);
      lighting_pass.end();

      const lighting_start = new Date();
      device.queue.submit([lighting_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const lighting_end = new Date();

      
      const lightingTimeGPU = (lighting_end.getTime() - lighting_start.getTime())/1000;
      log(VB.TIME, "Render Canvas Time (GPU): ", lightingTimeGPU);
      break;
  }
  
}

async function dispatch_buffer_to_texture(key, texture) {
  const buffer = gpu_tensors[key].gpuBufferData;

  const shader_module = device.createShaderModule({ code: buffer_to_texture_shader_code });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader_module, entryPoint: "main" }
  });

  const bind_group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffer } },
      { binding: 1, resource: texture.createView() }
    ]
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind_group);
  pass.dispatchWorkgroups(Math.ceil(800/8), Math.ceil(800/8));
  pass.end();
  const timer_start = new Date();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const timer_end = new Date();

  const buf_to_tex_time = (timer_end.getTime() - timer_start.getTime())/1000;
  log(VB.TIME, "Buffer to Texture Time (GPU): ", buf_to_tex_time);
}