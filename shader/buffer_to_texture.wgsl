@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var outTex : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    if (gid.x >= 800u || gid.y >= 800u) { return; }

    let base = (gid.y * 800u + gid.x) * 3u;
    let r = src[base] * 0.5 + 0.5;
    let g = src[base + 1u] * 0.5 + 0.5;
    let b = src[base + 2u] * 0.5 + 0.5;

    textureStore(outTex, vec2<i32>(gid.xy), vec4<f32>(r, g, b, 1.0));
}