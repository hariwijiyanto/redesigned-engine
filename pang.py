import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import struct
from ecdsa import SECP256k1
from ecdsa.ellipticcurve import Point

# ==========================================
# 1. KONFIGURASI
# ==========================================
# GANTI DENGAN PUBLIC KEY TARGET ANDA (Compressed Hex)
PUBKEY_HEX = "02E96C3241F8C93475965413159042F3705C2E77626F76E5528340E7D3F6531988" 

# Logika Pencarian
START_EXPONENT = 8      # Mulai 2^8
TOTAL_SCALARS = 16      # Sampai 2^(8+16)
TRAP_SIZE = 500000      # Ukuran Trap Table
BLOOM_BITS = TRAP_SIZE * 20
BLOCK_SIZE = 128
GRID_SIZE = 1024
ITER_PER_THREAD = 200

# Curve Params
P_CURVE = SECP256k1.curve.p()
N_ORDER = SECP256k1.order
G_POINT = SECP256k1.generator

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def int_to_u32_array(val):
    arr = []
    for _ in range(8):
        arr.append(val & 0xFFFFFFFF)
        val >>= 32
    return np.array(arr, dtype=np.uint32)

def point_to_bytes(point):
    # Flatten struct: X(8), Y(8), Z(8), infinity(4 bytes bool + padding)
    # Total 100 bytes + padding = 128 bytes alignment ideal
    # Kita pack manual X, Y, Z (Z=1).
    buf = bytearray()
    buf.extend(int_to_u32_array(point.x()).tobytes())
    buf.extend(int_to_u32_array(point.y()).tobytes())
    buf.extend(int_to_u32_array(1).tobytes())
    buf.extend(struct.pack('I', 0)) # infinity=false
    # Padding to 128 bytes (32*4) agar alignment aman di array constant
    while len(buf) < 128:
        buf.append(0)
    return buf

def decompress_pubkey(pk_hex):
    x = int(pk_hex[2:], 16)
    y_sq = (pow(x, 3, P_CURVE) + 7) % P_CURVE
    y = pow(y_sq, (P_CURVE + 1) // 4, P_CURVE)
    if (y % 2) != (int(pk_hex[:2], 16) % 2): y = P_CURVE - y
    return Point(SECP256k1.curve, x, y)

# ==========================================
# 3. SETUP & PRECOMPUTE
# ==========================================
print("[*] Compiling CUDA Kernel...")
with open("peng.cu", "r") as f:
    mod = SourceModule(f.read(), no_extern_c=True)

# Load Constants
const_p = mod.get_global("const_p")[0]
const_G_jac = mod.get_global("const_G_jacobian")[0]
const_Precomputed = mod.get_global("const_PrecomputedPoints")[0]

cuda.memcpy_htod(const_p, int_to_u32_array(P_CURVE))
cuda.memcpy_htod(const_G_jac, point_to_bytes(G_POINT))

# Precompute Scalars P_i = 2^k * G
print(f"[*] Precomputing {TOTAL_SCALARS} points (Powers of 2)...")
full_scalars = [2**(START_EXPONENT + i) for i in range(TOTAL_SCALARS)]
precomputed_bytes = bytearray()
for s in full_scalars:
    P_res = G_POINT * s
    precomputed_bytes.extend(point_to_bytes(P_res))

cuda.memcpy_htod(const_Precomputed, precomputed_bytes)

# Upload Scalar Values (untuk reconstruction)
scalar_vals_gpu = cuda.mem_alloc(TOTAL_SCALARS * 8)
cuda.memcpy_htod(scalar_vals_gpu, np.array(full_scalars, dtype=np.uint64))

# ==========================================
# 4. GENERATE TRAP TABLE (GPU)
# ==========================================
print(f"[*] Generating Trap Table ({TRAP_SIZE} entries)...")
d_trap = cuda.mem_alloc(TRAP_SIZE * 16) # fp(8), idx(8)
bloom_ints = (BLOOM_BITS + 31) // 32
d_bloom = cuda.mem_alloc(bloom_ints * 4)
cuda.memset_d32(d_bloom, 0, bloom_ints)

gen_kernel = mod.get_function("generate_trap_table_kernel")
gen_kernel(d_trap, d_bloom, np.uint32(TRAP_SIZE), np.uint64(BLOOM_BITS),
           block=(128,1,1), grid=((TRAP_SIZE+127)//128, 1))
cuda.Context.synchronize()

# Download Trap Table ke RAM (HashMap)
print("[*] Copying Trap Table to Host for verification...")
trap_buf = bytearray(TRAP_SIZE * 16)
cuda.memcpy_dtoh(trap_buf, d_trap)
trap_dict = {}
for i in range(TRAP_SIZE):
    off = i * 16
    fp, idx = struct.unpack("QQ", trap_buf[off:off+16])
    trap_dict[fp] = idx
print(f"[*] Dictionary ready. Size: {len(trap_dict)}")

# ==========================================
# 5. START SEARCH
# ==========================================
target_point = decompress_pubkey(PUBKEY_HEX)
print(f"[*] Target: {target_point.x():x}")

d_target = cuda.mem_alloc(128) # struct size
cuda.memcpy_htod(d_target, point_to_bytes(target_point))

d_result = cuda.mem_alloc(32) # found, k_trap, sum, fp
search_kernel = mod.get_function("search_subset_sum_kernel")

print("[*] Running GPU Search...")
start_t = time.time()
counter = 0

while True:
    cuda.memset_d8(d_result, 0, 32)
    seed = int(time.time() * 1000000)
    
    search_kernel(
        d_target, d_bloom, d_result,
        np.uint64(BLOOM_BITS), np.uint64(seed),
        np.int32(ITER_PER_THREAD), np.int32(TOTAL_SCALARS),
        scalar_vals_gpu,
        block=(BLOCK_SIZE,1,1), grid=(GRID_SIZE,1)
    )
    cuda.Context.synchronize()

    # Check Result
    res = bytearray(32)
    cuda.memcpy_dtoh(res, d_result)
    found = struct.unpack("i", res[:4])[0]

    if found:
        _, _, n_step, fp = struct.unpack("iQQQ", res)
        print(f"\n[!] MATCH FOUND ON GPU! FP: {fp}")
        
        if fp in trap_dict:
            k_trap = trap_dict[fp]
            print(f"[+] Confirmed in Trap Table! Index: {k_trap}")
            print(f"[+] Scalar Sum (n_step): {n_step}")
            
            # Key Recovery: T_search = k_trap * G
            # T_search = Target - n_step * G
            # Target = (k_trap + n_step) * G
            priv_key = (k_trap + n_step) % N_ORDER
            print(f"\n[SUCCESS] PRIVATE KEY: {hex(priv_key)}")
            
            # Verify
            if (G_POINT * priv_key).x() == target_point.x():
                print("[Verified] Public Key matches.")
                break
        else:
            print("[-] False Positive (Bloom Collision). Continuing...")

    counter += BLOCK_SIZE * GRID_SIZE * ITER_PER_THREAD
    if counter % 10000000 == 0:
        elapsed = time.time() - start_t
        rate = counter / elapsed / 1000000
        print(f"\rSpeed: {rate:.2f} M ops/sec | Total: {counter}", end="")

