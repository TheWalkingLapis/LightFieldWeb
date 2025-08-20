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
@group(0) @binding(2) var texSampler : sampler;
@group(0) @binding(3) var<uniform> uniforms : LightingUniforms;

@fragment
fn fsMain(@builtin(position) pos : vec4f) -> @location(0) vec4f {
    let uv = pos.xy / vec2f(800.0, 800.0); // normalized coordinates
    let xyz = textureSample(rgbTex, texSampler, uv).xyz * 2.0 - 1.0;
    let rgb = textureSample(xyzTex, texSampler, uv).rgb * 2.0 - 1.0;

    let epsilon = 0.01;
    if (length(xyz) < epsilon) {
        discard;
    }

    let light_dir = normalize(uniforms.light_pos);
    let cam_pos = uniforms.cam_pos;

    let cam_to_world = xyz - cam_pos;
    let depth = (3.0 - length(cam_to_world)) / 3; // assume camera radius (1.5) to be at min/max depth

    return vec4f(vec3f(depth), 1.0);

    /*

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
    let normal = -normalize(cross(dx, dy));
    let diffuse = max(dot(normal, light_dir), 0.0);

    //return vec4f(vec3f(diffuse), 1.0);
    return vec4f(normal, 1.0);
    */

}