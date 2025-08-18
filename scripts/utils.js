function compareTensors(t1, t2) {
  if (t1.length !== t2.length) {
    throw new Error(`Tensor size mismatch: ${t1.length} vs ${t2.length}`);
  }
  let epsilon = 0.00001;
  let maxDiff = 0;
  let sumSq = 0;
  for (let i = 0; i < t1.length; i++) {
    const diff = Math.abs(t1[i] - t2[i]);
    if (diff > epsilon) {
      if (diff > maxDiff) maxDiff = diff;
      sumSq += diff * diff;
    }
  }
  const mse = sumSq / t1.length;
  return [maxDiff, mse];
}

function gpu_tensor_from_dims(key, dims) {
  let n = 4; // float = 4 byte
  dims.forEach(i => n *= i);

  const buffer = device.createBuffer({
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    size: Math.ceil(n / 16) * 16 /* align to 16 bytes */
  });
  const tensor = ort.Tensor.fromGpuBuffer(buffer, {
    dataType: 'float32',
    dims: dims
  });
  
  gpu_tensors[key] = tensor;

  return tensor;
}

function get_browser_name() {
  const ua = navigator.userAgent;
  let browser = "Unknown";
  if (/Edg/i.test(ua)) browser = "Edge";
  if (/Chrome/i.test(ua) && !/Chromium/i.test(ua)) browser = "Chrome";
  if (/Firefox/i.test(ua)) browser = "Firefox";
  if (/Safari/i.test(ua) && !/Chrome/i.test(ua)) browser = "Safari";
  return browser;
}

function get_platform() {
  const is_mobile = /Mobi|Android|iPhone|iPad|iPod|Windows Phone/i.test(navigator.userAgent);
  return is_mobile ? "Mobile" : "Desktop";
}

function log(verbosity, ...txt) {
  if (verbosity <= verbose_level) {
    log_func = (verbosity == VB.ERROR) ? console.error : console.log;
    log_func(...txt)
  }
}