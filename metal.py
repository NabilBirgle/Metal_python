import random
import metalcompute as mc
from array import array
# The metal kernel to run
kernel = """
#include <metal_stdlib>;
using namespace metal;
kernel void helloABC(const device float *A [[buffer(0)]],
                       const device float *B [[buffer(1)]],
                       device float *C [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    C[id] = A[id] + B[id];
}
"""
count = 1000
# Create a metal device to run kernel with
dev = mc.Device()
# Create a compiled kernel to run
helloABC = dev.kernel(kernel).function("helloABC")
# Make a buffer for the result
A = array('f',range(count)) # 0..9
B = array('f',range(count)) # 10...19

for i in range(count):
    A[i] = random.randint(0,99) // 10
    B[i] = random.randint(0,99) // 10

C = dev.buffer(count * 4) # 4 because a float32 needs 4 bytes
C_view = memoryview(C).cast('f')
# Run the kernel count times
helloABC(count, A, B, C)
# Print the second result which should be 1.0 * 11.0 = 11.0
for i in range(5):
    print(A[i], ' + ', B[i], ' = ', C_view[i])
print('...')
