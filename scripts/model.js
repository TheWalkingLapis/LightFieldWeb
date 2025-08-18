let Sampler;
let Embedder;
let R2LEngine;

let pts_tensor;
let embb_pts_tensor;
let gpu_tensors = {};

async function evaluate() {
  const pts = await sample(); // TODO use pts from gpubuffer if reshape is baked into embedder

  const start = new Date();
  await R2LEngine.run({ input: pts }, { rgb: gpu_tensors["rgb"], xyz: gpu_tensors["xyz"] });
  const end = new Date();

  const inferenceTime = (end.getTime() - start.getTime())/1000;
  log(VB.TIME, "R2L Inference Time: ", inferenceTime);

}

async function sample() {
  const c2w = camera.get_c2w_as_input();
  const c2w33 = new ort.Tensor('float32', c2w.map(row => row.slice(0, 3)).flat(), [3, 3]);
  const c2w13 = new ort.Tensor('float32', c2w.map(row => [row[3]]).flat(), [1, 3]);

  const sample_start = new Date();
  await Sampler.run({ origin: c2w33, direction: c2w13}, { pts: gpu_tensors["pts"] });
  const sample_end = new Date();

  const embb_start = new Date();
  await Embedder.run({ pts: gpu_tensors["pts"] }, { embbpts: gpu_tensors["embb_pts"] });
  const embb_end = new Date();

  const sampleInferenceTime = (sample_end.getTime() - sample_start.getTime())/1000;
  const embbInferenceTime = (embb_end.getTime() - embb_start.getTime())/1000;
  log(VB.TIME, "Sample Time: ", sampleInferenceTime);
  log(VB.TIME, "Embedding Time: ", embbInferenceTime);

  return gpu_tensors["embb_pts"].reshape([1, 312, 100, 100]);
}