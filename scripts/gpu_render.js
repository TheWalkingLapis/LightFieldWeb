let gpu_canvas_struct = {};

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

async function display_output_gpu(key) {
  const context = gpu_canvas_struct[key]["ctx"];
  const format = navigator.gpu.getPreferredCanvasFormat();
  const buffer = gpu_tensors[key].gpuBufferData;

  const texture = device.createTexture({
    size: [800, 800],
    format: "rgba8unorm",
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING
  });

  const cpy_wgslCode = `
  @group(0) @binding(0) var<storage, read> src : array<f32>;
  @group(0) @binding(1) var outTex : texture_storage_2d<rgba8unorm, write>;

  @compute @workgroup_size(8, 8)
  fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x >= 800u || gid.y >= 800u) { return; }

    let idx = gid.y * 800u + gid.x;
    let r = src[idx];
    let g = src[idx + 800u*800u];
    let b = src[idx + 2u*800u*800u];

    textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(r, g, b, 1.0));
  }
  `;
  const cpy_shaderModule = device.createShaderModule({ code: cpy_wgslCode });

  const cpy_pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: cpy_shaderModule, entryPoint: "main" }
  });

  const cpy_bindGroup = device.createBindGroup({
    layout: cpy_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffer } },
      { binding: 1, resource: texture.createView() }
    ]
  });

  const cpy_encoder = device.createCommandEncoder();
  const cpy_pass = cpy_encoder.beginComputePass();
  cpy_pass.setPipeline(cpy_pipeline);
  cpy_pass.setBindGroup(0, cpy_bindGroup);
  cpy_pass.dispatchWorkgroups(Math.ceil(800/8), Math.ceil(800/8));
  cpy_pass.end();
  device.queue.submit([cpy_encoder.finish()]);

  const render_shader = `
  // Vertex shader: outputs a fullscreen triangle/quad
  @vertex
  fn vsMain(@builtin(vertex_index) idx : u32) -> @builtin(position) vec4f {
    var pos = array<vec2f,6>(
      vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
      vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
    );
    return vec4f(pos[idx], 0.0, 1.0);
  }

  // Fragment shader: samples your rgba8unorm texture
  @group(0) @binding(0) var myTex : texture_2d<f32>;
  @group(0) @binding(1) var mySampler : sampler;

  @fragment
  fn fsMain(@builtin(position) pos : vec4f) -> @location(0) vec4f {
    let uv = pos.xy / vec2f(800.0, 800.0); // normalized coordinates
    return textureSample(myTex, mySampler, uv);
  }
  `;

  const render_shaderModule = device.createShaderModule({ code: render_shader });

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
      { binding: 0, resource: texture.createView() },
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

  device.queue.submit([render_encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}