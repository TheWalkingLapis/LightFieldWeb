let Sampler;
let Embedder;
let R2LEngine;

let pts_tensor;
let embb_pts_tensor;
let gpu_tensors = {};

async function evaluate() {
  const pts = await sample(); // TODO use pts from gpubuffer if reshape is baked into embedder

  const start = performance.now();
  await R2LEngine.run({ input: pts }, { rgb: gpu_tensors["rgb"], xyz: gpu_tensors["xyz"] });
  const end = performance.now();

  const inferenceTime = (end - start)/1000;
  log(VB.TIME, "R2L Inference Time: ", inferenceTime);

}

async function sample() {
  const c2w = camera.get_c2w_as_input();
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