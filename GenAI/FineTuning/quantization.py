import numpy as np

# Create some sample float values (like model weights)
x = np.array([-1.0,-0.75, -0.2, 0.0 , 0.3,0.8,1.0])

# We will define quantization range (int8)
q_min = -128
q_max = 127

# find min and max float data
x_min = x.min()
x_max = x.max()

# Scale factor
scale = (x_max - x_min) / (q_max - q_min)

# zero pointer (which float value maps to 0 in int8)
zero_point = 0

# Quantize
q = np.round((x-zero_point)/scale)

q = np.clip(q,q_min,q_max)

q = q.astype(np.int8)

print("Original float values:", x)
print("Quantized int8 values:", q)

# Dequantize
dequantized_x = (q - zero_point) * scale

print("Dequantized float values:", dequantized_x)

# error
error = np.abs(x-dequantized_x)
print("Absolute error:", error)
print("Mean absolute error:", error.mean())
