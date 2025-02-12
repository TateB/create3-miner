#include <cuda_runtime.h>

__constant__ unsigned long long RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

__device__ inline unsigned long long ROTL64(unsigned long long x, int y) {
    return (x << y) | (x >> (64 - y));
}

__device__ void keccak_f(unsigned long long *state) {
    int piln[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };

    unsigned long long C[5], D;
    for (int round = 0; round < 24; round++) {
        for (int i = 0; i < 5; i++) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }

        for (int i = 0; i < 5; i++) {
            D = C[(i + 4) % 5] ^ ROTL64(C[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) {
                state[j + i] ^= D;
            }
        }

        unsigned long long tmp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = piln[i];
            C[0] = state[j];
            state[j] = ROTL64(tmp, (i + 1) * (i + 2) / 2 % 64);
            tmp = C[0];
        }

        for (int j = 0; j < 25; j += 5) {
            unsigned long long t0 = state[j + 0], t1 = state[j + 1], t2 = state[j + 2], t3 = state[j + 3], t4 = state[j + 4];
            state[j + 0] ^= ~t1 & t2;
            state[j + 1] ^= ~t2 & t3;
            state[j + 2] ^= ~t3 & t4;
            state[j + 3] ^= ~t4 & t0;
            state[j + 4] ^= ~t0 & t1;
        }

        state[0] ^= RC[round];
    }
}

__device__ void keccak256(unsigned char *localInput, unsigned char *localOutput, unsigned int inputLength) {
    const unsigned int rsize = 136; // 1088 bits (136 bytes) for Keccak-256
    unsigned long long state[25] = {0};
    unsigned int i = 0;
    while (i < inputLength) {
        if (i + rsize <= inputLength) {
            for (unsigned int j = 0; j < rsize / 8; j++) {
                unsigned long long block = 0;
                for (int k = 0; k < 8; k++) {
                    block |= (unsigned long long)(localInput[i + j * 8 + k]) << (8 * k);
                }
                state[j] ^= block;
            }
            keccak_f(state);
            i += rsize;
        } else {
            // Handle the last block with padding
            unsigned char padded[rsize] = {0};
            for (unsigned int j = 0; j < inputLength - i; j++) {
                padded[j] = localInput[i + j];
            }
            padded[inputLength - i] = 0x01; // Padding start
            padded[rsize - 1] |= 0x80; // Padding end
            for (unsigned int j = 0; j < rsize / 8; j++) {
                unsigned long long block = 0;
                for (int k = 0; k < 8; k++) {
                    block |= (unsigned long long)(padded[j * 8 + k]) << (8 * k);
                }
                state[j] ^= block;
            }
            keccak_f(state);
            break;
        }
    }
    if (inputLength == 0) {
        unsigned char padded[rsize] = {0};
        padded[0] = 0x01;
        padded[rsize - 1] |= 0x80;
        for (unsigned int j = 0; j < rsize / 8; j++) {
            unsigned long long block = 0;
            for (int k = 0; k < 8; k++) {
                block |= (unsigned long long)(padded[j * 8 + k]) << (8 * k);
            }
            state[j] ^= block;
        }
        keccak_f(state);
    }
    // Write the output
    for (unsigned int j = 0; j < 32; j++) {
        localOutput[j] = (unsigned char)((state[j / 8] >> (8 * (j % 8))) & 0xFF);
    }
}

typedef union {
    unsigned char bytes[32];
    unsigned int words[8];
} SaltUnion;

__constant__ unsigned int mixConstant = 0x9e3779b9;

__constant__ unsigned char CREATE3_FACTORY[20] = {
    0x00, 0x4e, 0xe0, 0x12, 0xd7, 0x7c, 0x5d, 0x0e,
    0x67, 0xd8, 0x61, 0x04, 0x1d, 0x11, 0x82, 0x4f,
    0x51, 0xb5, 0x90, 0xfb
};

__constant__ unsigned char FACTORY_BYTECODE_HASH[32] = {
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 
    0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54, 0x2f, 
    0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 
    0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49, 0x7c, 0x1f
};

extern "C" __global__ void vanity_search(
    const unsigned char* deployer,
    const unsigned char* prefix,
    const unsigned int* prefix_len,
    const unsigned char* ns,
    const unsigned int* ns_len,
    const unsigned char* initial_salt,
    unsigned char* result_salt,
    int* found
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (atomicOr(found, 0) != 0) {
        return;
    }

    SaltUnion salt;

    for (unsigned int i = 0; i < 32; i++) {
        salt.bytes[i] = initial_salt[i];
    }
    for (unsigned int j = 0; j < 8; j++) {
        salt.words[j] ^= tid;
        salt.words[j] *= mixConstant;
    }

    // First compute namespaced salt
    unsigned char input[256];
    unsigned int input_len = 0;
    unsigned char final_salt[32];

    if (*ns_len > 0) {
        // Add salt
        for (unsigned int i = 0; i < 32; i++) {
            input[input_len++] = salt.bytes[i];
        }
        // Add namespace
        for (unsigned int i = 0; i < *ns_len; i++) {
            input[input_len++] = ns[i];
        }
        
        keccak256(input, final_salt, input_len);
    } else {
        for (unsigned int i = 0; i < 32; i++) {
            final_salt[i] = salt.bytes[i];
        }
    }

    // Then hash with deployer
    input_len = 0;
    for (unsigned int i = 0; i < 20; i++) {
        input[input_len++] = deployer[i];
    }
    for (unsigned int i = 0; i < 32; i++) {
        input[input_len++] = final_salt[i];
    }
    keccak256(input, final_salt, input_len);

    // Compute CREATE2 address for proxy
    input_len = 0;
    input[input_len++] = 0xff;  // CREATE2 prefix
    for (unsigned int i = 0; i < 20; i++) {
        input[input_len++] = CREATE3_FACTORY[i];
    }
    for (unsigned int i = 0; i < 32; i++) {
        input[input_len++] = final_salt[i];
    }
    for (unsigned int i = 0; i < 32; i++) {
        input[input_len++] = FACTORY_BYTECODE_HASH[i];
    }
    keccak256(input, final_salt, input_len);

    // Finally compute CREATE address for contract
    input_len = 0;
    input[input_len++] = 0xd6;  // RLP prefix
    input[input_len++] = 0x94;  // Address marker
    for (unsigned int i = 0; i < 20; i++) {
        input[input_len++] = final_salt[12 + i];
    }
    input[input_len++] = 0x01;  // Nonce
    keccak256(input, final_salt, input_len);

    // Compare the hex values with the prefix
    bool matches = true;
    for (unsigned int i = 0; i < *prefix_len && matches; i++) {
        // Get the byte from the address and extract the correct nibble
        unsigned int byte_index = i / 2;
        unsigned int nibble_index = i % 2;
        unsigned char address_byte = final_salt[12 + byte_index];
        unsigned char address_nibble;
        if (nibble_index == 0) {
            address_nibble = (address_byte >> 4) & 0xF;  // high nibble
        } else {
            address_nibble = address_byte & 0xF;  // low nibble
        }
        matches = matches && (prefix[i] == address_nibble);
    }
    
    if (matches) {
        if (atomicCAS(found, 0, 1) == 0) {
            for (unsigned int i = 0; i < 32; i++) {
                result_salt[i] = salt.bytes[i];
            }
            return;
        }
    }
} 