#include <cuda_runtime.h>

__constant__ unsigned long long RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

__device__ inline unsigned long long ROTL64(unsigned long long x, int y)
{
    return (x << y) | (x >> (64 - y));
}

__device__ const int rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};

__device__ const int piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};

__device__ void keccak_f(unsigned long long *state)
{
    unsigned long long t, bc[5];

    for (int round = 0; round < 24; round++)
    {
        // Theta
        bc[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        bc[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        bc[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        bc[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        bc[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        t = bc[4] ^ ROTL64(bc[1], 1);
        state[0] ^= t;
        state[5] ^= t;
        state[10] ^= t;
        state[15] ^= t;
        state[20] ^= t;

        t = bc[0] ^ ROTL64(bc[2], 1);
        state[1] ^= t;
        state[6] ^= t;
        state[11] ^= t;
        state[16] ^= t;
        state[21] ^= t;

        t = bc[1] ^ ROTL64(bc[3], 1);
        state[2] ^= t;
        state[7] ^= t;
        state[12] ^= t;
        state[17] ^= t;
        state[22] ^= t;

        t = bc[2] ^ ROTL64(bc[4], 1);
        state[3] ^= t;
        state[8] ^= t;
        state[13] ^= t;
        state[18] ^= t;
        state[23] ^= t;

        t = bc[3] ^ ROTL64(bc[0], 1);
        state[4] ^= t;
        state[9] ^= t;
        state[14] ^= t;
        state[19] ^= t;
        state[24] ^= t;

        // Rho Pi
        t = state[1];
        int j0 = piln[0];
        bc[0] = state[j0];
        state[j0] = ROTL64(t, rotc[0]);
        t = bc[0];
        int j1 = piln[1];
        bc[0] = state[j1];
        state[j1] = ROTL64(t, rotc[1]);
        t = bc[0];
        int j2 = piln[2];
        bc[0] = state[j2];
        state[j2] = ROTL64(t, rotc[2]);
        t = bc[0];
        int j3 = piln[3];
        bc[0] = state[j3];
        state[j3] = ROTL64(t, rotc[3]);
        t = bc[0];
        int j4 = piln[4];
        bc[0] = state[j4];
        state[j4] = ROTL64(t, rotc[4]);
        t = bc[0];
        int j5 = piln[5];
        bc[0] = state[j5];
        state[j5] = ROTL64(t, rotc[5]);
        t = bc[0];
        int j6 = piln[6];
        bc[0] = state[j6];
        state[j6] = ROTL64(t, rotc[6]);
        t = bc[0];
        int j7 = piln[7];
        bc[0] = state[j7];
        state[j7] = ROTL64(t, rotc[7]);
        t = bc[0];
        int j8 = piln[8];
        bc[0] = state[j8];
        state[j8] = ROTL64(t, rotc[8]);
        t = bc[0];
        int j9 = piln[9];
        bc[0] = state[j9];
        state[j9] = ROTL64(t, rotc[9]);
        t = bc[0];
        int j10 = piln[10];
        bc[0] = state[j10];
        state[j10] = ROTL64(t, rotc[10]);
        t = bc[0];
        int j11 = piln[11];
        bc[0] = state[j11];
        state[j11] = ROTL64(t, rotc[11]);
        t = bc[0];
        int j12 = piln[12];
        bc[0] = state[j12];
        state[j12] = ROTL64(t, rotc[12]);
        t = bc[0];
        int j13 = piln[13];
        bc[0] = state[j13];
        state[j13] = ROTL64(t, rotc[13]);
        t = bc[0];
        int j14 = piln[14];
        bc[0] = state[j14];
        state[j14] = ROTL64(t, rotc[14]);
        t = bc[0];
        int j15 = piln[15];
        bc[0] = state[j15];
        state[j15] = ROTL64(t, rotc[15]);
        t = bc[0];
        int j16 = piln[16];
        bc[0] = state[j16];
        state[j16] = ROTL64(t, rotc[16]);
        t = bc[0];
        int j17 = piln[17];
        bc[0] = state[j17];
        state[j17] = ROTL64(t, rotc[17]);
        t = bc[0];
        int j18 = piln[18];
        bc[0] = state[j18];
        state[j18] = ROTL64(t, rotc[18]);
        t = bc[0];
        int j19 = piln[19];
        bc[0] = state[j19];
        state[j19] = ROTL64(t, rotc[19]);
        t = bc[0];
        int j20 = piln[20];
        bc[0] = state[j20];
        state[j20] = ROTL64(t, rotc[20]);
        t = bc[0];
        int j21 = piln[21];
        bc[0] = state[j21];
        state[j21] = ROTL64(t, rotc[21]);
        t = bc[0];
        int j22 = piln[22];
        bc[0] = state[j22];
        state[j22] = ROTL64(t, rotc[22]);
        t = bc[0];
        int j23 = piln[23];
        bc[0] = state[j23];
        state[j23] = ROTL64(t, rotc[23]);
        t = bc[0];

        // Chi
        bc[1] = state[1];
        bc[2] = state[2];
        bc[3] = state[3];
        bc[4] = state[4];
        state[0] ^= (~bc[1]) & bc[2];
        state[1] ^= (~bc[2]) & bc[3];
        state[2] ^= (~bc[3]) & bc[4];
        state[3] ^= (~bc[4]) & bc[0];
        state[4] ^= (~bc[0]) & bc[1];

        bc[1] = state[6];
        bc[2] = state[7];
        bc[3] = state[8];
        bc[4] = state[9];
        state[5] ^= (~bc[1]) & bc[2];
        state[6] ^= (~bc[2]) & bc[3];
        state[7] ^= (~bc[3]) & bc[4];
        state[8] ^= (~bc[4]) & bc[0];
        state[9] ^= (~bc[0]) & bc[1];

        bc[1] = state[11];
        bc[2] = state[12];
        bc[3] = state[13];
        bc[4] = state[14];
        state[10] ^= (~bc[1]) & bc[2];
        state[11] ^= (~bc[2]) & bc[3];
        state[12] ^= (~bc[3]) & bc[4];
        state[13] ^= (~bc[4]) & bc[0];
        state[14] ^= (~bc[0]) & bc[1];

        bc[1] = state[16];
        bc[2] = state[17];
        bc[3] = state[18];
        bc[4] = state[19];
        state[15] ^= (~bc[1]) & bc[2];
        state[16] ^= (~bc[2]) & bc[3];
        state[17] ^= (~bc[3]) & bc[4];
        state[18] ^= (~bc[4]) & bc[0];
        state[19] ^= (~bc[0]) & bc[1];

        bc[1] = state[21];
        bc[2] = state[22];
        bc[3] = state[23];
        bc[4] = state[24];
        state[20] ^= (~bc[1]) & bc[2];
        state[21] ^= (~bc[2]) & bc[3];
        state[22] ^= (~bc[3]) & bc[4];
        state[23] ^= (~bc[4]) & bc[0];
        state[24] ^= (~bc[0]) & bc[1];

        // Iota
        state[0] ^= RC[round];
    }
}

__device__ void keccak256(unsigned char *input, unsigned char *output, unsigned int inputLength)
{
    const unsigned int rsize = 136; // 1088 bits (136 bytes) for Keccak-256
    unsigned long long state[25] = {0};
    unsigned int i;

    // Absorb input
    for (i = 0; i + rsize <= inputLength; i += rsize)
    {
        for (unsigned int j = 0; j < rsize / 8; j++)
        {
            state[j] ^= ((unsigned long long *)&input[i])[j];
        }
        keccak_f(state);
    }

    // Handle remaining input and padding
    unsigned char last_block[rsize] = {0};
    unsigned int remaining = inputLength - i;
    if (remaining > 0)
    {
        for (unsigned int j = 0; j < remaining; j++)
        {
            last_block[j] = input[i + j];
        }
    }
    last_block[remaining] = 0x01;
    last_block[rsize - 1] |= 0x80;

    for (unsigned int j = 0; j < rsize / 8; j++)
    {
        state[j] ^= ((unsigned long long *)last_block)[j];
    }

    keccak_f(state);

    // Extract output
    for (unsigned int j = 0; j < 32; j++)
    {
        output[j] = ((unsigned char *)state)[j];
    }
}

typedef union
{
    unsigned char bytes[32];
    unsigned int words[8];
} SaltUnion;

__constant__ unsigned int mixConstant = 0x9e3779b9;

extern "C" __global__ void vanity_search(
    const unsigned char *deployer,
    const unsigned char *prefix,
    const unsigned int *prefix_len,
    const unsigned char *bytecode_hash,
    const unsigned char *initial_salt,
    unsigned char *result_salt,
    int *found)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (atomicOr(found, 0) != 0)
    {
        return;
    }

    SaltUnion salt;

    for (unsigned int i = 0; i < 32; i++)
    {
        salt.bytes[i] = initial_salt[i];
    }
    for (unsigned int j = 0; j < 8; j++)
    {
        salt.words[j] ^= tid + j;
        salt.words[j] *= mixConstant;
    }

    // First compute namespaced salt
    unsigned char input[256];
    unsigned int input_len = 0;
    unsigned char final_salt[32];

    for (unsigned int i = 0; i < 32; i++)
    {
        final_salt[i] = salt.bytes[i];
    }

    // Compute CREATE2 address
    input_len = 0;
    input[input_len++] = 0xff; // CREATE2 prefix
    for (unsigned int i = 0; i < 20; i++)
    {
        input[input_len++] = deployer[i];
    }
    for (unsigned int i = 0; i < 32; i++)
    {
        input[input_len++] = final_salt[i];
    }
    for (unsigned int i = 0; i < 32; i++)
    {
        input[input_len++] = bytecode_hash[i];
    }
    keccak256(input, final_salt, input_len);

    // Compare the hex values with the prefix
    bool matches = true;
    for (unsigned int i = 0; i < *prefix_len && matches; i++)
    {
        // Get the byte from the address and extract the correct nibble
        unsigned int byte_index = i / 2;
        unsigned int nibble_index = i % 2;
        unsigned char address_byte = final_salt[12 + byte_index];
        unsigned char address_nibble;
        if (nibble_index == 0)
        {
            address_nibble = (address_byte >> 4) & 0xF; // high nibble
        }
        else
        {
            address_nibble = address_byte & 0xF; // low nibble
        }
        matches = matches && (prefix[i] == address_nibble);
    }

    if (matches)
    {
        if (atomicCAS(found, 0, 1) == 0)
        {
            for (unsigned int i = 0; i < 32; i++)
            {
                result_salt[i] = salt.bytes[i];
            }
            return;
        }
    }
}