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
    pad: f32,
    cam_pos : vec3<f32>,
    _pad: f32
};

// Fragment shader: samples your rgba8unorm texture
@group(0) @binding(0) var rgbTex : texture_2d<f32>;
@group(0) @binding(1) var xyzTex : texture_2d<f32>;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var maskTex : texture_2d<f32>;
@group(0) @binding(4) var texSampler : sampler;
@group(0) @binding(5) var<uniform> uniforms : LightingUniforms;

@fragment
fn fsMain(@builtin(position) pos : vec4f) -> @location(0) vec4f {
    let uv = pos.xy / vec2f(800.0, 800.0); // normalized coordinates
    let rgb = textureSample(rgbTex, texSampler, uv).rgb * 2.0 - 1.0;
    let xyz = textureSample(xyzTex, texSampler, uv).rgb;// * 2.0 - 1.0;
    let normal = textureSample(normalTex, texSampler, uv).rgb * 2.0 - 1.0;
    let mask = textureSample(maskTex, texSampler, uv).r * 2.0 - 1.0;

    let epsilon = 0.01;
    if (length(xyz) < epsilon) {
        discard;
    }

    let light_dir = normalize(uniforms.light_pos);
    let cam_pos = uniforms.cam_pos;

    let cam_to_object = xyz - cam_pos;
    let depth = (length(cam_to_object) - 1) / 2; // world pos in [-.5, .5] -> depth in [1, 2]

    //return vec4f(vec3f(depth), 1.0);
    let diffuse = max(dot(normal, light_dir), 0.0) * rgb;
    let ambient = vec3f(1.0, 1.0, 1.0) * 0.1;

    if (mask < 0.001) {
        //return vec4f(1.0, 1.0, 1.0, 1.0);    
        //if (dot(-normalize(cam_pos), light_dir) > 0.8) {
        //    return vec4f(0.0, 0.0, 0.0, 1.0);
        //}
        return vec4f(1.0, 1.0, 1.0, 1.0);
    }

    return vec4f((ambient + diffuse) * mask, 1.0);

}