
// NOTE: This is identical to the other reduce shader, except here we first multiply
// the samples with a weight texture. (For hemicube-hemisphere conversion and custom material
// properties)

// Mostly from: https://therealmjp.github.io/posts/average-luminance-compute-shader/

static const uint THREAD_GROUP_SIZE = 8;
static const uint NUM_THREADS_IN_GROUP = THREAD_GROUP_SIZE * THREAD_GROUP_SIZE;
groupshared float4 shared_mem[NUM_THREADS_IN_GROUP];

Texture2D<float4> input_tex : register(t0, space0);
Texture2D<float2> weight_tex : register(t1, space0);
RWTexture2D<float4> output_tex : register(u0, space1);

cbuffer Uniforms : register(b0, space2)
{
    uint2 tex_offset;  // Offset into the texture, in pixels.
}

// NOTE: The alpha component is important, it contains a weighted count of invalid samples.

// TODO: Check if this is actually correct, lightmapper.h seems to be doing something different here.

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID)
{
    const uint thread_id = group_thread_id.y * THREAD_GROUP_SIZE + group_thread_id.x;

    const uint2 sample_idx = tex_offset + (group_id.xy * THREAD_GROUP_SIZE + group_thread_id.xy) * 2;
    float4 s0 = input_tex[sample_idx + uint2(0, 0)];
    float4 s1 = input_tex[sample_idx + uint2(1, 0)];
    float4 s2 = input_tex[sample_idx + uint2(0, 1)];
    float4 s3 = input_tex[sample_idx + uint2(1, 1)];
    float2 w0 = weight_tex[sample_idx + uint2(0, 0)];
    float2 w1 = weight_tex[sample_idx + uint2(1, 0)];
    float2 w2 = weight_tex[sample_idx + uint2(0, 1)];
    float2 w3 = weight_tex[sample_idx + uint2(1, 1)];
    float4 s = float4(s0.rgb * w0.r, s0.a * w0.g) +
               float4(s1.rgb * w1.r, s1.a * w1.g) +
               float4(s2.rgb * w2.r, s2.a * w2.g) +
               float4(s3.rgb * w3.r, s3.a * w3.g);

    // Store sample in shared memory.
    shared_mem[thread_id] = s;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction.
    [unroll(NUM_THREADS_IN_GROUP)]
    for(uint i = NUM_THREADS_IN_GROUP / 2; i > 0; i >>= 1)
    {
        if(thread_id < i)
            shared_mem[thread_id] += shared_mem[thread_id + i];

        GroupMemoryBarrierWithGroupSync();
    }

    // The first thread writes the final value.
    if(thread_id == 0)
        output_tex[tex_offset + group_id.xy] = shared_mem[0];
}
