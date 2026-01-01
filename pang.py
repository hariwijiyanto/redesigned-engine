import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import sys
from ecdsa import SECP256k1

# --- KONFIGURASI ---
PUBKEY_HEX = "02..." # MASUKKAN PUBLIC KEY TARGET DISINI (Compressed)
TRAP_SIZE = 1000000  # 1 Juta entri trap
BLOOM_BITS = TRAP_SIZE * 20 # Ukuran bloom filter (~2.5 MB)
BLOCK_SIZE = 256
GRID_SIZE = 2048     # Jumlah blok
ITER_PER_THREAD = 100 # Loop per thread kernel

# SECP256k1 Constants
P = SECP256k1.curve.p()
N = SECP256k1.order
G_POINT = SECP256k1.generator

def hex_to_bigint_array(hex_str):
    """Convert hex string to uint32 array (8 words) for GPU."""
    val = int(hex_str, 16)
    arr = []
    for _ in range(8):
        arr.append(val & 0xFFFFFFFF)
        val >>= 32
    return np.array(arr, dtype=np.uint32)

def int_to_bigint_array(val):
    arr = []
    for _ in range(8):
        arr.append(val & 0xFFFFFFFF)
        val >>= 32
    return np.array(arr, dtype=np.uint32)

def create_jac_point(point):
    """Convert affine point to Jacobian (X, Y, 1) struct format."""
    x_arr = int_to_bigint_array(point.x())
    y_arr = int_to_bigint_array(point.y())
    z_arr = int_to_bigint_array(1)
    
    # Structure: X[8], Y[8], Z[8], infinity (32bit bool/pad)
    # Alignment note: Cuda struct alignment is tricky using numpy structs.
    # Safe way: flatten logic or precise offsets.
    # Simple way here: Pass pointers or linearized array.
    # Let's use simple struct packing via numpy dtype.
    return (x_arr, y_arr, z_arr)

# Load CUDA Code
print("[*] Compiling CUDA Kernel...")
with open("peng.cu", "r") as f:
    cuda_code = f.read()

mod = SourceModule(cuda_code, no_extern_c=True)

# Get Kernels
gen_trap_kernel = mod.get_function("generate_trap_table_kernel_bigint")
precompute_G = mod.get_function("precompute_G_table_kernel")
search_kernel = mod.get_function("search_kernel")

# Set Constants
const_p = mod.get_global("const_p")[0]
const_n = mod.get_global("const_n")[0]
const_G_jac = mod.get_global("const_G_jacobian")[0]

cuda.memcpy_htod(const_p, int_to_bigint_array(P))
cuda.memcpy_htod(const_n, int_to_bigint_array(N))

# G Jacobian constant init
g_x = int_to_bigint_array(G_POINT.x())
g_y = int_to_bigint_array(G_POINT.y())
g_z = int_to_bigint_array(1)
# Struct packing for G: 8*4 + 8*4 + 8*4 + 4 (bool align) = 100 bytes approx
# We construct byte buffer manually to match struct ECPointJac
g_struct = bytearray()
g_struct.extend(g_x.tobytes())
g_struct.extend(g_y.tobytes())
g_struct.extend(g_z.tobytes())
g_struct.extend(b'\x00\x00\x00\x00') # infinity = false
cuda.memcpy_htod(const_G_jac, g_struct)

# Run Precompute
print("[*] Precomputing G Table on GPU...")
precompute_G(block=(256,1,1), grid=(1,1))
cuda.Context.synchronize()

# --- STEP 1: GENERATE TRAP TABLE ---
print(f"[*] Generating Trap Table ({TRAP_SIZE} entries)...")

# TrapEntry: uint64 fp, uint64 index
trap_struct_size = 16 
trap_mem_size = TRAP_SIZE * trap_struct_size
d_trap_table = cuda.mem_alloc(trap_mem_size)

# Bloom Filter
bloom_ints = (BLOOM_BITS + 31) // 32
d_bloom = cuda.mem_alloc(bloom_ints * 4)
cuda.memset_d32(d_bloom, 0, bloom_ints)

# Launch Gen Kernel
block_dim = (BLOCK_SIZE, 1, 1)
grid_dim = ((TRAP_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE, 1)
gen_trap_kernel(d_trap_table, d_bloom, np.uint32(TRAP_SIZE), np.uint64(BLOOM_BITS),
                block=block_dim, grid=grid_dim)
cuda.Context.synchronize()

# Copy Trap Table to Host (for verification later)
print("[*] Copying Trap Table to Host for caching...")
# We only need this if a hit occurs, but keeping a dict in RAM is good for lookups.
# For optimization, we can map FP -> Index on CPU.
# Read Trap Table buffer
trap_buffer = bytearray(trap_mem_size)
cuda.memcpy_dtoh(trap_buffer, d_trap_table)

trap_dict = {}
# Parse buffer (fp: offset 0, index: offset 8)
import struct
for i in range(TRAP_SIZE):
    off = i * 16
    fp, idx = struct.unpack("QQ", trap_buffer[off:off+16])
    trap_dict[fp] = idx

print(f"[*] Trap Table ready. Dict size: {len(trap_dict)}")

# --- STEP 2: SEARCH ---
# Prepare Target Point
# Parse Target Pubkey (compressed)
from ecdsa.ellipticcurve import Point
def decompress_pubkey(pk_hex):
    x = int(pk_hex[2:], 16)
    y_sq = (pow(x, 3, P) + 7) % P
    y = pow(y_sq, (P + 1) // 4, P)
    if (y % 2) != (int(pk_hex[:2], 16) % 2): y = P - y
    return Point(SECP256k1.curve, x, y)

target_point = decompress_pubkey(PUBKEY_HEX)
print(f"[*] Target Point: {target_point.x()}, {target_point.y()}")

# Struct Result
# struct SearchResult { int found; uint64_t k_trap; uint64_t n_step; uint64_t fp_match; };
res_struct_size = 32 # 4 + 4(pad) + 8 + 8 + 8
d_result = cuda.mem_alloc(res_struct_size)
cuda.memset_d8(d_result, 0, res_struct_size)

# Construct Target Point Jacobian struct bytes
t_x = int_to_bigint_array(target_point.x())
t_y = int_to_bigint_array(target_point.y())
t_z = int_to_bigint_array(1)
t_bytes = bytearray()
t_bytes.extend(t_x.tobytes())
t_bytes.extend(t_y.tobytes())
t_bytes.extend(t_z.tobytes())
t_bytes.extend(b'\x00'*4) # infinity false

print("[*] Starting GPU Search...")
start_time = time.time()
counter = 0

try:
    while True:
        seed_offset = int(time.time() * 1000) + counter
        
        # Args: target_jac (by value/bytes), d_bloom, d_result, bloom_size, seed, iter
        # Note: Struct by value in pycuda needs precise handling.
        # Easiest is to adjust kernel to accept pointers for struct, 
        # BUT since we pass it once, passing bytes as parameter works if kernel signature matches.
        # Kernel: (ECPointJac target, ...) -> ECPointJac is 3*32 + 4 bytes = 100 bytes.
        # PyCUDA might struggle passing large struct by value.
        # BETTER STRATEGY: Copy target to __constant__ or Global memory once.
        # Let's use Global Memory for target point for safety.
        
        d_target = cuda.mem_alloc(len(t_bytes))
        cuda.memcpy_htod(d_target, t_bytes)
        
        # CHANGE KERNEL SIGNATURE IN YOUR MIND: 
        # __global__ void search_kernel(ECPointJac* target_ptr, ...)
        # I will update the kernel signature in memory call below assuming pointer
        
        search_kernel(d_target, d_bloom, d_result, np.uint64(BLOOM_BITS), np.uint64(seed_offset), np.int32(ITER_PER_THREAD),
                      block=(BLOCK_SIZE,1,1), grid=(GRID_SIZE,1))
        
        cuda.Context.synchronize()
        
        # Check result
        res_buf = bytearray(res_struct_size)
        cuda.memcpy_dtoh(res_buf, d_result)
        found, _, k_trap, n_step, fp_match = struct.unpack("iiQQQ", res_buf)
        
        if found:
            print(f"\n[!!!] FOUND SIGNAL ON GPU!")
            print(f"FP Match: {fp_match}")
            print(f"Checking CPU Trap Table...")
            
            if fp_match in trap_dict:
                real_k_trap = trap_dict[fp_match]
                print(f"Confirmed in Trap Table! Index: {real_k_trap}")
                
                # Recover Key: K = k_trap + n_step (approx, check sign)
                # T_search = T - n_step*G. If match trap (k*G), then k*G = T - n_step*G => T = (k+n_step)G
                final_key = (real_k_trap + n_step) % N
                print(f"Candidate Private Key: {hex(final_key)}")
                
                # Verification
                if (G_POINT * final_key).x() == target_point.x():
                    print("SUCCESS: Private Key Verified!")
                    with open("FOUND_KEY.txt", "w") as f:
                        f.write(hex(final_key))
                    break
                else:
                    print("False positive (collision). Continuing...")
                    # Reset found flag
                    cuda.memset_d8(d_result, 0, res_struct_size)
            else:
                print("Bloom filter false positive (Not in dict). Continuing...")
                cuda.memset_d8(d_result, 0, res_struct_size)

        counter += BLOCK_SIZE * GRID_SIZE * ITER_PER_THREAD
        if counter % 1000000 == 0:
            elapsed = time.time() - start_time
            rate = counter / elapsed
            sys.stdout.write(f"\rSpeed: {rate:,.0f} keys/sec | Total: {counter:,}")
            sys.stdout.flush()

except KeyboardInterrupt:
    print("\nStopped.")
