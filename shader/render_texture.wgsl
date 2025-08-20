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
    let value = textureSample(myTex, mySampler, uv);
    return vec4f(vec3f(value.xyz) * 2.0 - 1.0, 1.0);
}