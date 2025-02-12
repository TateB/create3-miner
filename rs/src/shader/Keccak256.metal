#include <metal_stdlib>
using namespace metal;

constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

inline ulong ROTL64(ulong x, int y) {
    return (x << y) | (x >> (64 - y));
}

void keccak_f(thread ulong *state) {
    int piln[24] = { 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1 };

    ulong C[5], D;
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

        ulong tmp = state[1];
        for (int i = 0; i < 24; i++) {
            int j = piln[i];
            C[0] = state[j];
            state[j] = ROTL64(tmp, (i + 1) * (i + 2) / 2 % 64);
            tmp = C[0];
        }

        for (int j = 0; j < 25; j += 5) {
            ulong t0 = state[j + 0], t1 = state[j + 1], t2 = state[j + 2], t3 = state[j + 3], t4 = state[j + 4];
            state[j + 0] ^= ~t1 & t2;
            state[j + 1] ^= ~t2 & t3;
            state[j + 2] ^= ~t3 & t4;
            state[j + 3] ^= ~t4 & t0;
            state[j + 4] ^= ~t0 & t1;
        }

        state[0] ^= RC[round];
    }
}

void keccak256(thread uchar *localInput, thread uchar *localOutput, thread uint inputLength)  {
    const uint rsize = 136; // 1088 bits (136 bytes) for Keccak-256
    thread ulong state[25] = {0};
    thread uint i = 0;
    while (i < inputLength) {
        if (i + rsize <= inputLength) {
            for (uint j = 0; j < rsize / 8; j++) {
                thread ulong block = 0;
                for (int k = 0; k < 8; k++) {
                    block |= (ulong)(localInput[i + j * 8 + k]) << (8 * k);
                }
                state[j] ^= block;
            }
            keccak_f(state);
            i += rsize;
        } else {
            // Handle the last block with padding
            uchar padded[rsize] = {0};
            for (uint j = 0; j < inputLength - i; j++) {
                padded[j] = localInput[i + j];
            }
            padded[inputLength - i] = 0x01; // Padding start
            padded[rsize - 1] |= 0x80; // Padding end
            for (uint j = 0; j < rsize / 8; j++) {
                thread ulong block = 0;
                for (int k = 0; k < 8; k++) {
                    block |= (ulong)(padded[j * 8 + k]) << (8 * k);
                }
                state[j] ^= block;
            }
            keccak_f(state);
            break;
        }
    }
    if (inputLength == 0) {
        uchar padded[rsize] = {0};
        padded[0] = 0x01;
        padded[rsize - 1] |= 0x80;
        for (uint j = 0; j < rsize / 8; j++) {
            thread ulong block = 0;
            for (int k = 0; k < 8; k++) {
                block |= (ulong)(padded[j * 8 + k]) << (8 * k);
            }
            state[j] ^= block;
        }
        keccak_f(state);
    }
    // Write the output
    for (uint j = 0; j < 32; j++) {
        localOutput[j] = (uchar)((state[j / 8] >> (8 * (j % 8))) & 0xFF);
    }
}

typedef union {
    uchar  bytes[32];
    uint   words[8];
} SaltUnion;

constant uint mixConstant = 0x9e3779b9;

constant uchar CREATE3_FACTORY[20] = {
    0x00, 0x4e, 0xe0, 0x12, 0xd7, 0x7c, 0x5d, 0x0e,
    0x67, 0xd8, 0x61, 0x04, 0x1d, 0x11, 0x82, 0x4f,
    0x51, 0xb5, 0x90, 0xfb
};

constant uchar FACTORY_BYTECODE_HASH[32] = {
    0x21, 0xc3, 0x5d, 0xbe, 0x1b, 0x34, 0x4a, 0x24, 
    0x88, 0xcf, 0x33, 0x21, 0xd6, 0xce, 0x54, 0x2f, 
    0x8e, 0x9f, 0x30, 0x55, 0x44, 0xff, 0x09, 0xe4, 
    0x99, 0x3a, 0x62, 0x31, 0x9a, 0x49, 0x7c, 0x1f
};

kernel void vanity_search(
    const device uchar* deployer [[buffer(0)]],
    const device uchar* prefix [[buffer(1)]],
    const device uint* prefix_len [[buffer(2)]],
    const device uchar* ns [[buffer(3)]],
    const device uint* ns_len [[buffer(4)]],
    const device uchar* initial_salt [[buffer(5)]],
    device uchar* result_salt [[buffer(6)]],
    device atomic_bool* found [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (atomic_load_explicit(found, memory_order_relaxed)) {
        return;
    }

    thread SaltUnion salt;

    for (uint i = 0; i < 32; i++) {
        salt.bytes[i] = initial_salt[i];
    }
    for (uint j = 0; j < 8; j++) {
        salt.words[j] ^= tid;
        salt.words[j] *= mixConstant;
    }

    // First compute namespaced salt
    thread uchar input[256];
    thread uint input_len = 0;
    thread uchar final_salt[32];

    if (*ns_len > 0) {
        // Add salt
        for (uint i = 0; i < 32; i++) {
            input[input_len++] = salt.bytes[i];
        }
        // Add namespace
        for (uint i = 0; i < *ns_len; i++) {
            input[input_len++] = ns[i];
        }
        
        keccak256(input, final_salt, input_len);
    } else {
        for (uint i = 0; i < 32; i++) {
            final_salt[i] = salt.bytes[i];
        }
    }

    // Then hash with deployer
    input_len = 0;
    for (uint i = 0; i < 20; i++) {
        input[input_len++] = deployer[i];
    }
    for (uint i = 0; i < 32; i++) {
        input[input_len++] = final_salt[i];
    }
    keccak256(input, final_salt, input_len);

    // Compute CREATE2 address for proxy
    input_len = 0;
    input[input_len++] = 0xff;  // CREATE2 prefix
    for (uint i = 0; i < 20; i++) {
        input[input_len++] = CREATE3_FACTORY[i];
    }
    for (uint i = 0; i < 32; i++) {
        input[input_len++] = final_salt[i];
    }
    for (uint i = 0; i < 32; i++) {
        input[input_len++] = FACTORY_BYTECODE_HASH[i];
    }
    keccak256(input, final_salt, input_len);

    // Finally compute CREATE address for contract
    input_len = 0;
    input[input_len++] = 0xd6;  // RLP prefix
    input[input_len++] = 0x94;  // Address marker
    for (uint i = 0; i < 20; i++) {
        input[input_len++] = final_salt[12 + i];
    }
    input[input_len++] = 0x01;  // Nonce
    keccak256(input, final_salt, input_len);

    // Compare the hex values with the prefix
    bool matches = true;
    for (uint i = 0; i < *prefix_len && matches; i++) {
        // Get the byte from the address and extract the correct nibble
        uint byte_index = i / 2;
        uint nibble_index = i % 2;
        uchar address_byte = final_salt[12 + byte_index];
        uchar address_nibble;
        if (nibble_index == 0) {
            address_nibble = (address_byte >> 4) & 0xF;  // high nibble
        } else {
            address_nibble = address_byte & 0xF;  // low nibble
        }
        matches = matches && (prefix[i] == address_nibble);
    }
    
    if (matches) {
        bool expected = false;
        if (atomic_compare_exchange_weak_explicit(found, &expected, true, memory_order_relaxed, memory_order_relaxed)) {
            for (uint i = 0; i < 32; i++) {
                result_salt[i] = salt.bytes[i];
            }
            return;
        }
    }

    // atomic_fetch_add_explicit(debug_counter, 1, memory_order_relaxed);
}