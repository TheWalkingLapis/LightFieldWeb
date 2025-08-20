let gpu_canvas_struct = {};
let intermediate_gpu_textures = {};
let webgpu_command_display_texture = {};
let webgpu_command_buffer_to_texture = {};

let buffer_to_texture_shader_code;
let render_texture_shader_code;
let lighting_shader_code;

async function init_gpu_render() {
  buffer_to_texture_shader_code = await fetch("./shader/buffer_to_texture.wgsl").then(r => r.text());
  render_texture_shader_code = await fetch("./shader/render_texture.wgsl").then(r => r.text());
  lighting_shader_code = await fetch("./shader/lighting.wgsl").then(r => r.text());

  await create_gpu_canvas("rgb");
  await create_gpu_canvas("xyz");
  await create_gpu_canvas("lighting");

  const gpu_canvas_div = document.getElementById("gpuCanvasDiv");
  const lighting_canvas_div = document.getElementById("lightingCanvasDiv");

  lighting_canvas_div.appendChild(gpu_canvas_struct["lighting"]["ctx"].canvas);
  gpu_canvas_div.appendChild(gpu_canvas_struct["rgb"]["ctx"].canvas);
  gpu_canvas_div.appendChild(gpu_canvas_struct["xyz"]["ctx"].canvas);

  await create_gpu_intermediate_texture("rgb");
  await create_gpu_intermediate_texture("xyz");

  {
    const shader_module = device.createShaderModule({ code: buffer_to_texture_shader_code });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shader_module, entryPoint: "main" }
    });
    webgpu_command_buffer_to_texture["pipeline"] = pipeline;
    webgpu_command_buffer_to_texture["bindGroupLayout"] = pipeline.getBindGroupLayout(0);
  }
  {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const shaderModule = device.createShaderModule({ code: render_texture_shader_code });
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: shaderModule, entryPoint: "vsMain" },
      fragment: {
        module: shaderModule,
        entryPoint: "fsMain",
        targets: [{ format }]
      },
      primitive: { topology: "triangle-list" }
    });
    webgpu_command_display_texture["pipeline"] = pipeline;
    const sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear"
    });
    webgpu_command_display_texture["sampler"] = sampler;
    webgpu_command_display_texture["bindGroupLayout"] = pipeline.getBindGroupLayout(0);
  }
}

async function create_gpu_intermediate_texture(key) {
  const tex = device.createTexture({
    size: [800, 800],
    format: "rgba8unorm",
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
  });
  intermediate_gpu_textures[key] = tex;
}

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

      const display_texture = intermediate_gpu_textures[key];

      await dispatch_buffer_to_texture(key, display_texture);

      const render_pipeline = webgpu_command_display_texture["pipeline"];
      const render_sampler = webgpu_command_display_texture["sampler"];

      const render_bindGroup = device.createBindGroup({
        layout: webgpu_command_display_texture["bindGroupLayout"],
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

      const render_start = performance.now();
      device.queue.submit([render_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const render_end = performance.now();

      
      const renderTimeGPU = (render_end - render_start)/1000;
      log(VB.TIME, "Render Canvas Time (GPU): ", renderTimeGPU);
      break;


    case RENDER_MODES.LIGHTING:
      const lighting_context = gpu_canvas_struct[key]["ctx"];

      const rgb_texture = intermediate_gpu_textures["rgb"];
      const xyz_texture = intermediate_gpu_textures["xyz"];

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

      const cam_pos = camera.get_position();
      log(VB.INFO, "camera position:", cam_pos);
      const lighting_uniform_data = new Float32Array([0.0, 10.0, 0.0, 0.0, cam_pos[0], cam_pos[1], cam_pos[2], 0.0]); // uinforms including padding

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

      const lighting_start = performance.now();
      device.queue.submit([lighting_encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const lighting_end = performance.now();

      
      const lightingTimeGPU = (lighting_end - lighting_start)/1000;
      log(VB.TIME, "Render Canvas Time (GPU): ", lightingTimeGPU);
      break;
  }
  
}

async function dispatch_buffer_to_texture(key, texture) {
  const buffer = gpu_tensors[key].gpuBufferData;

  const pipeline = webgpu_command_buffer_to_texture["pipeline"];

  const bind_group = device.createBindGroup({
    layout: webgpu_command_buffer_to_texture["bindGroupLayout"],
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
  const timer_start = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const timer_end = performance.now();

  const buf_to_tex_time = (timer_end - timer_start)/1000;
  log(VB.TIME, "Buffer to Texture Time (GPU): ", buf_to_tex_time);
}