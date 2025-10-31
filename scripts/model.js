let Sampler;
let Embedder;
let R2LEngine;

let pts_tensor;
let embb_pts_tensor;
let gpu_tensors = {};


async function evaluate(c2w_path="") {
  const pts = await sample(c2w_path); // TODO use pts from gpubuffer if reshape is baked into embedder

  const start = performance.now();
  await R2LEngine.run({ input: pts }, { rgb: gpu_tensors["rgb"], xyz: gpu_tensors["xyz"], normal: gpu_tensors["normal"], mask: gpu_tensors["mask"] });
  const end = performance.now();

  const inferenceTime = (end - start)/1000;
  log(VB.TIME, "R2L Inference Time: ", inferenceTime);
  times["eval"].push(inferenceTime);
}

async function loadC2W(url) {
  const response = await fetch(url);
  const mat = await response.json();
  return mat;
}

async function sample(c2w_path="") {
  const c2w = c2w_path == "" ? camera.get_c2w_as_input() : await loadC2W(c2w_path);
  camera.c2w_to_position(c2w);
  const c2w33 = new ort.Tensor('float32', c2w.map(row => row.slice(0, 3)).flat(), [3, 3]);
  const c2w13 = new ort.Tensor('float32', c2w.map(row => [row[3]]).flat(), [1, 3]);

  const sample_start = performance.now();
  await Sampler.run({ origin: c2w33, direction: c2w13}, { pts: gpu_tensors["pts"] });
  const sample_end = performance.now();

  const embb_start = performance.now();
  await Embedder.run({ pts: gpu_tensors["pts"] }, { embbpts: gpu_tensors["embb_pts"] });
  const embb_end = performance.now();

  const sampleInferenceTime = (sample_end - sample_start)/1000;
  const embbInferenceTime = (embb_end - embb_start)/1000;
  log(VB.TIME, "Sample Time: ", sampleInferenceTime);
  log(VB.TIME, "Embedding Time: ", embbInferenceTime);

  return gpu_tensors["embb_pts"].reshape([1, 312, 100, 100]);
}

async function save_outputs() {
  async function save_tensor_as_bin(key, filename) {
    const data = await gpu_tensors[key].toData();
    const blob = new Blob([data.buffer], { type: "application/octet-stream" });

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
  }
  save_tensor_as_bin("rgb", "rgb.bin")
  save_tensor_as_bin("xyz", "xyz.bin")
  save_tensor_as_bin("normal", "normal.bin")
  save_tensor_as_bin("mask", "mask.bin")
}