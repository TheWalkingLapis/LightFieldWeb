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
  await create_gpu_canvas("normal");
  await create_gpu_canvas("mask");
  await create_gpu_canvas("lighting");

  const gpu_canvas_div = document.getElementById("gpuCanvasDiv");
  const lighting_canvas_div = document.getElementById("lightingCanvasDiv");

  lighting_canvas_div.appendChild(gpu_canvas_struct["lighting"]["ctx"].canvas);
  gpu_canvas_div.appendChild(gpu_canvas_struct["rgb"]["ctx"].canvas);
  gpu_canvas_div.appendChild(gpu_canvas_struct["xyz"]["ctx"].canvas);
  gpu_canvas_div.appendChild(gpu_canvas_struct["normal"]["ctx"].canvas);
  gpu_canvas_div.appendChild(gpu_canvas_struct["mask"]["ctx"].canvas);

  await create_gpu_intermediate_texture("rgb");
  await create_gpu_intermediate_texture("xyz");
  await create_gpu_intermediate_texture("normal");
  await create_gpu_intermediate_texture("mask");

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

async function copy_png_to_texture(key, texture, tag, index, gt=true) {
  let path = "pt/" + tag + "_gen_images/" + index.toString().padStart(3, '0') + "_"
  switch(key) {
    case "rgb": 
      path += "rgb";
      break;
    case "xyz": 
      path += "xyz";
      break;
    case "normal": 
      path += "n";
      break;
    case "mask": 
      path += "mask";
      break;
  }
  if(gt) {
    path += "_gt";
  }
  path += ".png";

  console.log(path)
  const resp = await fetch(path);
  const blob = await resp.blob();
  const bitmap = await createImageBitmap(blob);

  device.queue.copyExternalImageToTexture(
    { source: bitmap },                        
    { texture: texture, origin: { x: 0, y: 0, z: 0 } }, 
    { width: bitmap.width, height: bitmap.height, depthOrArrayLayers: 1 }
  );
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
  const [channels, height, width] = [key == "mask" ? 1 : 3, (key == "lighting" ? 800 : 400), (key == "lighting" ? 800 : 400)];

  const canvas = document.createElement("canvas");
  canvas.id = key + "_gpu";
  canvas.width = width;
  canvas.height = height;
  if (canvas_callbacks) {
    connect_to_canvas(camera, light_source, canvas);
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

async function display_output_gpu(key = "", tag="project_lego", index=0, gt=true) {
  
  const format = navigator.gpu.getPreferredCanvasFormat();

  switch (key) {
    case "lighting":
      const lighting_context = gpu_canvas_struct[key]["ctx"];

      const rgb_texture = intermediate_gpu_textures["rgb"];
      const xyz_texture = intermediate_gpu_textures["xyz"];
      const normal_texture = intermediate_gpu_textures["normal"];
      const mask_texture = intermediate_gpu_textures["mask"];

      await copy_png_to_texture("rgb", rgb_texture, tag, index, gt);
      await copy_png_to_texture("xyz", xyz_texture, tag, index), gt;
      await copy_png_to_texture("normal", normal_texture, tag, index, gt);
      await copy_png_to_texture("mask", mask_texture, tag, index, gt);

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
      const light_pos = light_source.get_position();
      log(VB.INFO, "camera position:", cam_pos);
      log(VB.INFO, "light position:", light_pos);
      const lighting_uniform_data = new Float32Array([light_pos[0], light_pos[1], light_pos[2], 0.0, cam_pos[0], cam_pos[1], cam_pos[2], 0.0]); // uinforms including padding

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
          { binding: 2, resource: normal_texture.createView() },
          { binding: 3, resource: mask_texture.createView() },
          { binding: 4, resource: lighting_sampler },
          { binding: 5, resource: { buffer: lighting_uniform_buffer} }
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
      
    default:
      const context = gpu_canvas_struct[key]["ctx"];

      const display_texture = intermediate_gpu_textures[key];

      await copy_png_to_texture(key, display_texture, tag, index, gt);

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


  }
  
}

async function dispatch_buffer_to_texture(key, texture) {
  const buffer = gpu_tensors[key].gpuBufferData;

  const pipeline = webgpu_command_buffer_to_texture["pipeline"];

  const isMaskData = new Uint32Array([key == "mask" ? 1 : 0]);

  const isMaskBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(isMaskBuffer, 0, isMaskData.buffer);

  const bind_group = device.createBindGroup({
    layout: webgpu_command_buffer_to_texture["bindGroupLayout"],
    entries: [
      { binding: 0, resource: { buffer: buffer } },
      { binding: 1, resource: texture.createView() },
      { binding: 2, resource: {buffer: isMaskBuffer } }
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