// Vertex shader: outputs a fullscreen triangle/quad
@vertex
fn vsMain(@builtin(vertex_index) idx : u32) -> @builtin(position) vec4f {
    var pos = array<vec2f,6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
        vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0)
    );
    return vec4f(pos[idx], 0.0, 1.0);
}

struct LightingUniforms {
    light_pos : vec3<f32>,   
    _pad  : f32,         // padding for 16-byte alignment
};

// Fragment shader: samples your rgba8unorm texture
@group(0) @binding(0) var rgbTex : texture_2d<f32>;
@group(0) @binding(1) var xyzTex : texture_2d<f32>;
@group(0) @binding(2) var texSampler : sampler;
@group(0) @binding(3) var<uniform> uniforms : LightingUniforms;

@fragment
fn fsMain(@builtin(position) pos : vec4f) -> @location(0) vec4f {
    let uv = pos.xy / vec2f(800.0, 800.0); // normalized coordinates
    let xyz = textureSample(rgbTex, texSampler, uv).xyz;
    let rgb = textureSample(xyzTex, texSampler, uv).rgb;

    let light_dir = normalize(uniforms.light_pos);

    let texSize = vec2f(800.0, 800.0);
    let offsetX = vec2<f32>(1.0 / texSize.x, 0.0);
    let offsetY = vec2<f32>(0.0, 1.0 / texSize.y);

    // Sample neighboring positions
    let posR = textureSample(xyzTex, texSampler, uv + offsetX).xyz;
    let posU = textureSample(xyzTex, texSampler, uv + offsetY).xyz;

    // Compute screen-space derivatives
    let dx = posR - xyz;
    let dy = posU - xyz;

    //let dx = dpdxFine(xyz);
    //let dy = dpdyFine(xyz);
    let normal = -normalize(cross(dy, dx));
    let diffuse = max(dot(normal, light_dir), 0.0);

    //return vec4f(vec3f(diffuse), 1.0);
    return vec4f(diffuse * rgb, 1.0);
}