# blur_conv_webgpu_fixed.py
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np

# Model parameters
height, width = 800, 800
channels = 3
kernel_size = 3

# Create a 3-channel blur kernel (no groups)
# Each channel gets the same averaging kernel
weights = np.zeros((channels, channels, kernel_size, kernel_size), dtype=np.float32)
for c in range(channels):
    weights[c, c] = 1.0 / (kernel_size * kernel_size)

# Bias
bias = np.zeros((channels,), dtype=np.float32)

# ONNX initializers
weight_initializer = numpy_helper.from_array(weights, name="conv_weight")
bias_initializer = numpy_helper.from_array(bias, name="conv_bias")

# Input/output tensors (fixed shape)
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, channels, height, width])
output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, channels, height, width])

# Conv node (no groups)
conv_node = helper.make_node(
    "Conv",
    inputs=["input", "conv_weight", "conv_bias"],
    outputs=["output"],
    kernel_shape=[kernel_size, kernel_size],
    pads=[1, 1, 1, 1],  # same padding
    strides=[1, 1],
)

# Graph
graph = helper.make_graph(
    [conv_node],
    "BlurConvWebGPU",
    [input_tensor],
    [output_tensor],
    initializer=[weight_initializer, bias_initializer]
)

# Model
model = helper.make_model(graph, producer_name="blur_conv_webgpu", opset_imports=[helper.make_operatorsetid("", 13)])

# Save
onnx.save(model, "blur_conv_webgpu.onnx")
print("Saved WebGPU-compatible blur model: 'blur_conv_webgpu.onnx'")