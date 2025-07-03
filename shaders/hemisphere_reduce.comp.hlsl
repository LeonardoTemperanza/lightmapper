
// Mostly from: https://therealmjp.github.io/posts/average-luminance-compute-shader/

static const uint THREAD_GROUP_SIZE = 8;
static const uint NUM_THREADS = THREAD_GROUP_SIZE * THREAD_GROUP_SIZE;
groupshared float4 shared_mem[NUM_THREADS];

Texture2D<float4> input_tex : register(t0, space0);
RWTexture2D<float4> output_tex : register(u0, space1);

cbuffer Uniforms : register(b0, space2)
{
    uint2 tex_offset;  // Offset into the texture, in pixels.
}

// NOTE: The alpha component is important, it contains a weighted count of invalid samples.

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID)
{
    const uint thread_id = group_thread_id.y * THREAD_GROUP_SIZE + group_thread_id.x;

    const uint2 sample_idx = tex_offset + (group_id.xy * THREAD_GROUP_SIZE + group_thread_id.xy) * 2;
    float4 s = 0.0f;
    s += input_tex[sample_idx + uint2(0, 0)];
    s += input_tex[sample_idx + uint2(1, 0)];
    s += input_tex[sample_idx + uint2(0, 1)];
    s += input_tex[sample_idx + uint2(1, 1)];

    // Store sample in shared memory.
    shared_mem[thread_id] = s;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction.
    [unroll(NUM_THREADS)]
    for(uint s = NUM_THREADS / 2; s > 0; s >>= 1)
    {
        if(thread_id < s)
            shared_mem[thread_id] += shared_mem[thread_id + s];

        GroupMemoryBarrierWithGroupSync();
    }

    // The first thread writes the final value.
    if(thread_id == 0)
        output_tex[tex_offset + group_id.xy] = shared_mem[0];
}
