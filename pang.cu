#include <cuda_runtime.h>
#include <stdint.h>

#define BIGINT_WORDS 8
#define MAX_PRECOMPUTED_POINTS 32 

// --- STRUKTUR DATA ---
struct BigInt {
    uint32_t data[BIGINT_WORDS];
};

struct ECPointJac {
    BigInt X, Y, Z;
    bool infinity;
};

struct ECPoint {
    BigInt x, y;
    bool infinity;
};

struct SearchResult {
    int found;                  // 1 jika ditemukan
    uint64_t k_trap;            // Index dari Trap Table
    uint64_t n_step_scalar_sum; // Total nilai skalar yang ditemukan
    uint64_t fp_match;          // Fingerprint
};

struct TrapEntry {
    uint64_t fp;
    uint64_t index;
};

// --- VARIABLE CONSTANT ---
__constant__ BigInt const_p;
__constant__ ECPointJac const_G_jacobian;
__constant__ ECPointJac const_PrecomputedPoints[MAX_PRECOMPUTED_POINTS];

// --- FUNGSI MATH DASAR (ASM OPTIMIZED) ---

__device__ __forceinline__ void init_bigint(BigInt *x, uint32_t val) {
    x->data[0] = val;
    #pragma unroll
    for (int i = 1; i < BIGINT_WORDS; i++) x->data[i] = 0;
}

__device__ __forceinline__ void copy_bigint(BigInt *dest, const BigInt *src) {
    #pragma unroll
    for (int i = 0; i < BIGINT_WORDS; i++) dest->data[i] = src->data[i];
}

__device__ __forceinline__ int compare_bigint(const BigInt *a, const BigInt *b) {
    for (int i = BIGINT_WORDS - 1; i >= 0; i--) {
        if (a->data[i] > b->data[i]) return 1;
        if (a->data[i] < b->data[i]) return -1;
    }
    return 0;
}

__device__ __forceinline__ void ptx_u256Add(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile (
        "add.cc.u32 %0, %8, %16; \n\t"
        "addc.cc.u32 %1, %9, %17; \n\t"
        "addc.cc.u32 %2, %10, %18; \n\t"
        "addc.cc.u32 %3, %11, %19; \n\t"
        "addc.cc.u32 %4, %12, %20; \n\t"
        "addc.cc.u32 %5, %13, %21; \n\t"
        "addc.cc.u32 %6, %14, %22; \n\t"
        "addc.u32    %7, %15, %23; \n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__device__ __forceinline__ void ptx_u256Sub(BigInt *res, const BigInt *a, const BigInt *b) {
    asm volatile (
        "sub.cc.u32 %0, %8, %16; \n\t"
        "subc.cc.u32 %1, %9, %17; \n\t"
        "subc.cc.u32 %2, %10, %18; \n\t"
        "subc.cc.u32 %3, %11, %19; \n\t"
        "subc.cc.u32 %4, %12, %20; \n\t"
        "subc.cc.u32 %5, %13, %21; \n\t"
        "subc.cc.u32 %6, %14, %22; \n\t"
        "subc.u32    %7, %15, %23; \n\t"
        : "=r"(res->data[0]), "=r"(res->data[1]), "=r"(res->data[2]), "=r"(res->data[3]),
          "=r"(res->data[4]), "=r"(res->data[5]), "=r"(res->data[6]), "=r"(res->data[7])
        : "r"(a->data[0]), "r"(a->data[1]), "r"(a->data[2]), "r"(a->data[3]),
          "r"(a->data[4]), "r"(a->data[5]), "r"(a->data[6]), "r"(a->data[7]),
          "r"(b->data[0]), "r"(b->data[1]), "r"(b->data[2]), "r"(b->data[3]),
          "r"(b->data[4]), "r"(b->data[5]), "r"(b->data[6]), "r"(b->data[7])
    );
}

__device__ __forceinline__ void sub_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    BigInt temp;
    if (compare_bigint(a, b) < 0) {
         BigInt sum;
         ptx_u256Add(&sum, a, &const_p);
         ptx_u256Sub(&temp, &sum, b);
    } else {
         ptx_u256Sub(&temp, a, b);
    }
    copy_bigint(res, &temp);
}

// Implementasi MulMod Sederhana (Barrett reduction idealnya, ini versi basic slow-but-correct)
// Untuk kecepatan di kernel search, kita HINDARI mul_mod.
// Kernel search hanya pakai ADD/SUB, jadi mul_mod hanya untuk pre-calc atau konversi akhir.
__device__ void mul_mod_device(BigInt *res, const BigInt *a, const BigInt *b) {
    // Versi Placeholder: Implementasi penuh Montgomery/Barrett butuh baris kode banyak.
    // Disini kita gunakan asumsi dasar. Jika stuck, gunakan kode lama untuk fungsi ini.
    // Untuk kode ini, KITA TIDAK MEMAKAI mul_mod di SEARCH LOOP (Optimasi Kunci).
    // Jadi fungsi ini hanya dipanggil saat konversi Jacobian->Affine.
    
    // Implementasi sangat sederhana (Slow shift-add) cukup untuk konversi akhir.
    BigInt R; init_bigint(&R, 0);
    BigInt B_curr; copy_bigint(&B_curr, b);
    
    for(int i=0; i<256; i++) {
        int word = i >> 5; int bit = i & 31;
        if ((a->data[word] >> bit) & 1) {
            BigInt sum; ptx_u256Add(&sum, &R, &B_curr);
            if (compare_bigint(&sum, &const_p) >= 0) ptx_u256Sub(&R, &sum, &const_p);
            else copy_bigint(&R, &sum);
        }
        BigInt dbl; ptx_u256Add(&dbl, &B_curr, &B_curr);
        if (compare_bigint(&dbl, &const_p) >= 0) ptx_u256Sub(&B_curr, &dbl, &const_p);
        else copy_bigint(&B_curr, &dbl);
    }
    copy_bigint(res, &R);
}

// --- ARITMATIKA TITIK (JACOBIAN) ---

__device__ void point_set_infinity_jac(ECPointJac *P) {
    P->infinity = true;
    init_bigint(&P->X, 0); init_bigint(&P->Y, 0); init_bigint(&P->Z, 0);
}

__device__ void point_copy_jac(ECPointJac *dest, const ECPointJac *src) {
    copy_bigint(&dest->X, &src->X);
    copy_bigint(&dest->Y, &src->Y);
    copy_bigint(&dest->Z, &src->Z);
    dest->infinity = src->infinity;
}

__device__ void double_point_jac(ECPointJac *R, const ECPointJac *P) {
    if (P->infinity) { point_set_infinity_jac(R); return; }
    
    BigInt A, B, C, D, X3, Y3, Z3, temp;
    mul_mod_device(&A, &P->Y, &P->Y);
    mul_mod_device(&B, &P->X, &A); mul_mod_device(&B, &B, &B); // 4*X*Y^2 (approx)
    // Perbaiki logika double point standar:
    // S = 4*x*y^2. M = 3*x^2. 
    BigInt Y2, XY2, S, M, T;
    mul_mod_device(&Y2, &P->Y, &P->Y);
    mul_mod_device(&XY2, &P->X, &Y2);
    // S = 4 * XY2
    ptx_u256Add(&S, &XY2, &XY2); 
    if(compare_bigint(&S, &const_p)>=0) ptx_u256Sub(&S, &S, &const_p);
    ptx_u256Add(&S, &S, &S); 
    if(compare_bigint(&S, &const_p)>=0) ptx_u256Sub(&S, &S, &const_p);

    // M = 3 * X^2
    mul_mod_device(&T, &P->X, &P->X);
    ptx_u256Add(&M, &T, &T);
    if(compare_bigint(&M, &const_p)>=0) ptx_u256Sub(&M, &M, &const_p);
    ptx_u256Add(&M, &M, &T);
    if(compare_bigint(&M, &const_p)>=0) ptx_u256Sub(&M, &M, &const_p);
    
    // X3 = M^2 - 2S
    mul_mod_device(&X3, &M, &M);
    BigInt S2; ptx_u256Add(&S2, &S, &S);
    if(compare_bigint(&S2, &const_p)>=0) ptx_u256Sub(&S2, &S2, &const_p);
    sub_mod_device(&X3, &X3, &S2);
    
    // Y3 = M(S - X3) - 8*Y^4
    sub_mod_device(&T, &S, &X3);
    mul_mod_device(&Y3, &M, &T);
    mul_mod_device(&T, &Y2, &Y2); // Y^4
    // 8 * Y^4
    BigInt T2; ptx_u256Add(&T2, &T, &T); if(compare_bigint(&T2, &const_p)>=0) ptx_u256Sub(&T2, &T2, &const_p); //2
    ptx_u256Add(&T2, &T2, &T2); if(compare_bigint(&T2, &const_p)>=0) ptx_u256Sub(&T2, &T2, &const_p); //4
    ptx_u256Add(&T2, &T2, &T2); if(compare_bigint(&T2, &const_p)>=0) ptx_u256Sub(&T2, &T2, &const_p); //8
    sub_mod_device(&Y3, &Y3, &T2);
    
    // Z3 = 2*Y*Z
    mul_mod_device(&Z3, &P->Y, &P->Z);
    ptx_u256Add(&Z3, &Z3, &Z3);
    if(compare_bigint(&Z3, &const_p)>=0) ptx_u256Sub(&Z3, &Z3, &const_p);

    copy_bigint(&R->X, &X3); copy_bigint(&R->Y, &Y3); copy_bigint(&R->Z, &Z3);
    R->infinity = false;
}

__device__ void add_point_jac(ECPointJac *R, const ECPointJac *P, const ECPointJac *Q) {
    if (P->infinity) { point_copy_jac(R, Q); return; }
    if (Q->infinity) { point_copy_jac(R, P); return; }

    BigInt Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, r, V;
    mul_mod_device(&Z1Z1, &P->Z, &P->Z);
    mul_mod_device(&Z2Z2, &Q->Z, &Q->Z);
    mul_mod_device(&U1, &P->X, &Z2Z2);
    mul_mod_device(&U2, &Q->X, &Z1Z1);
    
    BigInt S1_t, S2_t;
    mul_mod_device(&S1_t, &P->Y, &Q->Z); mul_mod_device(&S1_t, &S1_t, &Z2Z2);
    mul_mod_device(&S2_t, &Q->Y, &P->Z); mul_mod_device(&S2_t, &S2_t, &Z1Z1);
    
    if (compare_bigint(&U1, &U2) == 0) {
        if (compare_bigint(&S1_t, &S2_t) == 0) {
            double_point_jac(R, P);
            return;
        } else {
            point_set_infinity_jac(R);
            return;
        }
    }
    
    sub_mod_device(&H, &U2, &U1);
    sub_mod_device(&r, &S2_t, &S1_t);
    mul_mod_device(&I, &H, &H); // I = H^2
    mul_mod_device(&J, &H, &I); // J = H^3
    mul_mod_device(&V, &U1, &I); // V = U1 * H^2
    
    // X3 = r^2 - J - 2V
    mul_mod_device(&R->X, &r, &r);
    sub_mod_device(&R->X, &R->X, &J);
    BigInt V2; ptx_u256Add(&V2, &V, &V); if(compare_bigint(&V2, &const_p)>=0) ptx_u256Sub(&V2, &V2, &const_p);
    sub_mod_device(&R->X, &R->X, &V2);
    
    // Y3 = r(V - X3) - 2*S1*J
    sub_mod_device(&R->Y, &V, &R->X);
    mul_mod_device(&R->Y, &R->Y, &r);
    BigInt S1J; mul_mod_device(&S1J, &S1_t, &J);
    BigInt S1J2; ptx_u256Add(&S1J2, &S1J, &S1J); if(compare_bigint(&S1J2, &const_p)>=0) ptx_u256Sub(&S1J2, &S1J2, &const_p);
    sub_mod_device(&R->Y, &R->Y, &S1J2);
    
    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H
    BigInt ZSum; ptx_u256Add(&ZSum, &P->Z, &Q->Z); if(compare_bigint(&ZSum, &const_p)>=0) ptx_u256Sub(&ZSum, &ZSum, &const_p);
    BigInt ZSumSq; mul_mod_device(&ZSumSq, &ZSum, &ZSum);
    sub_mod_device(&ZSumSq, &ZSumSq, &Z1Z1);
    sub_mod_device(&ZSumSq, &ZSumSq, &Z2Z2);
    mul_mod_device(&R->Z, &ZSumSq, &H);
    
    R->infinity = false;
}

// Fermat Inverse: a^(p-2)
__device__ void mod_inverse(BigInt *res, const BigInt *a) {
    BigInt p_minus_2, two;
    init_bigint(&two, 2);
    ptx_u256Sub(&p_minus_2, &const_p, &two);
    
    // ModExp Sederhana
    BigInt result; init_bigint(&result, 1);
    BigInt base; copy_bigint(&base, a);
    
    for (int i = 0; i < 256; i++) {
         if ((p_minus_2.data[i/32] >> (i%32)) & 1) {
              mul_mod_device(&result, &result, &base);
         }
         mul_mod_device(&base, &base, &base);
    }
    copy_bigint(res, &result);
}

__device__ void jacobian_to_affine(ECPoint *R, const ECPointJac *P) {
    if (P->infinity) { R->infinity = true; return; }
    BigInt Zinv, Zinv2, Zinv3;
    mod_inverse(&Zinv, &P->Z);
    mul_mod_device(&Zinv2, &Zinv, &Zinv);
    mul_mod_device(&Zinv3, &Zinv2, &Zinv);
    mul_mod_device(&R->x, &P->X, &Zinv2);
    mul_mod_device(&R->y, &P->Y, &Zinv3);
    R->infinity = false;
}

// --- UTILITIES: RNG & BLOOM FILTER ---

struct rng_state { uint64_t s[2]; };

__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ void init_rng(rng_state *state, uint64_t seed) {
    state->s[0] = splitmix64(seed);
    state->s[1] = splitmix64(seed + 1);
}

__device__ uint64_t xoroshiro128plus(rng_state *state) {
    const uint64_t s0 = state->s[0];
    uint64_t s1 = state->s[1];
    const uint64_t result = s0 + s1;
    s1 ^= s0;
    state->s[0] = ((s0 << 24) | (s0 >> 40)) ^ s1 ^ (s1 << 16);
    state->s[1] = (s1 << 37) | (s1 >> 27);
    return result;
}

__device__ bool check_bloom(const uint32_t* bloom, uint64_t bloom_bits, uint64_t fp) {
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);
    for (int j = 0; j < 2; j++) {
        uint64_t idx = (h1 + j * h2) % bloom_bits;
        if (!((bloom[idx/32] >> (idx%32)) & 1)) return false;
    }
    return true;
}

extern "C" {

// --- KERNEL 1: Generate Trap Table ---
__global__ void generate_trap_table_kernel(
    TrapEntry* d_trap_table,
    uint32_t* d_bloom,
    uint32_t trap_size,
    uint64_t bloom_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= trap_size) return;

    // Trap Value: k_trap = idx + 1
    // Point: P = (idx + 1) * G
    // Kita gunakan Double-and-Add sederhana dengan G
    BigInt k; init_bigint(&k, idx + 1);
    ECPointJac P; point_set_infinity_jac(&P);
    ECPointJac TempG; copy_bigint(&TempG.X, &const_G_jacobian.X); 
                      copy_bigint(&TempG.Y, &const_G_jacobian.Y); 
                      copy_bigint(&TempG.Z, &const_G_jacobian.Z); 
                      TempG.infinity = false;

    for(int i=0; i<32; i++) { // Asumsi trap size < 2^32
        if((k.data[0] >> i) & 1) add_point_jac(&P, &P, &TempG);
        double_point_jac(&TempG, &TempG);
    }

    ECPoint Aff; jacobian_to_affine(&Aff, &P);
    
    // Fingerprint
    uint64_t low64 = Aff.x.data[0] | ((uint64_t)Aff.x.data[1] << 32);
    uint64_t parity = (Aff.y.data[0] & 1);
    uint64_t fp = splitmix64(low64 ^ parity);

    d_trap_table[idx].fp = fp;
    d_trap_table[idx].index = idx + 1;

    // Bloom Insert
    uint64_t h1 = splitmix64(fp);
    uint64_t h2 = splitmix64(h1);
    for(int j=0; j<2; j++) {
        uint64_t bit_idx = (h1 + j * h2) % bloom_bits;
        atomicOr(&d_bloom[bit_idx/32], (1 << (bit_idx%32)));
    }
}

// --- KERNEL 2: Search (Sparse Subset Sum) ---
__global__ void search_subset_sum_kernel(
    ECPointJac* d_target_jac,
    uint32_t* d_bloom,
    SearchResult* d_result,
    uint64_t bloom_bits,
    uint64_t seed_offset,
    int iter_per_thread,
    int total_scalars,
    uint64_t* d_scalar_values
) {
    if (d_result->found) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    rng_state rng;
    init_rng(&rng, seed_offset + tid);
    
    ECPointJac Target; point_copy_jac(&Target, d_target_jac);

    for(int i=0; i<iter_per_thread; i++) {
        ECPointJac Accum; point_set_infinity_jac(&Accum);
        uint64_t sum_scalars = 0;
        
        // Randomly select subset
        // Kita gunakan 1 bit random per precomputed point (50% probability)
        // Atau sesuaikan dengan logic Python (random per group).
        // Disini kita implementasi: pure random subset mask.
        
        uint64_t mask = xoroshiro128plus(&rng); 
        
        for(int j=0; j<total_scalars; j++) {
            if((mask >> j) & 1) {
                add_point_jac(&Accum, &Accum, &const_PrecomputedPoints[j]);
                sum_scalars += d_scalar_values[j];
            }
        }
        
        if (Accum.infinity) continue;

        // T_search = Target - Accum
        // Negate Accum.Y
        sub_mod_device(&Accum.Y, &const_p, &Accum.Y);
        ECPointJac T_search;
        add_point_jac(&T_search, &Target, &Accum);
        
        // Convert & Check
        ECPoint Aff; jacobian_to_affine(&Aff, &T_search);
        uint64_t low64 = Aff.x.data[0] | ((uint64_t)Aff.x.data[1] << 32);
        uint64_t parity = (Aff.y.data[0] & 1);
        uint64_t fp = splitmix64(low64 ^ parity);
        
        if (check_bloom(d_bloom, bloom_bits, fp)) {
            if (atomicCAS(&d_result->found, 0, 1) == 0) {
                d_result->fp_match = fp;
                d_result->n_step_scalar_sum = sum_scalars;
            }
            return;
        }
    }
}

}
