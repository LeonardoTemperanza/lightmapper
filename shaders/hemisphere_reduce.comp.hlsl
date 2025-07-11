
// Mostly from: https://therealmjp.github.io/posts/average-luminance-compute-shader/

static const uint THREAD_GROUP_SIZE = 8;
static const uint NUM_THREADS = THREAD_GROUP_SIZE * THREAD_GROUP_SIZE;
groupshared float4 shared_mem[NUM_THREADS];

Texture2D<float4> input_tex : register(t0, space0);
RWTexture2D<float4> output_tex : register(u0, space1);

cbuffer Uniforms : register(b0, space2)
{
    uint2 input_size;
    uint2 output_size;  // NOTE: This is expected to be equal to the number of dispatched groups.
}

// NOTE: The alpha component is important, it contains a weighted count of invalid samples.

[numthreads(THREAD_GROUP_SIZE, THREAD_GROUP_SIZE, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID)
{
    const uint2 num_groups = output_size;
    const uint2 num_samples_per_thread = uint2(2, 2);
    const uint2 num_used_threads_per_group = input_size / num_groups / num_samples_per_thread;
    //const uint2 num_used_threads_per_group = uint2(8, 8);

    const uint local_thread_id = group_thread_id.y * THREAD_GROUP_SIZE + group_thread_id.x;

    const uint2 sample_idx = (num_used_threads_per_group * group_id.xy + group_thread_id.xy) * num_samples_per_thread;

    float4 s = 0.0f;
    if(all(group_thread_id.xy < num_used_threads_per_group))
    {
        s += input_tex[sample_idx + uint2(0, 0)];
        s += input_tex[sample_idx + uint2(1, 0)];
        s += input_tex[sample_idx + uint2(0, 1)];
        s += input_tex[sample_idx + uint2(1, 1)];
    }

    // Store sample in shared memory.
    shared_mem[local_thread_id] = s;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction.
    [unroll(NUM_THREADS)]
    for(uint s = NUM_THREADS / 2; s > 0; s >>= 1)
    {
        if(local_thread_id < s)
            shared_mem[local_thread_id] += shared_mem[local_thread_id + s];

        GroupMemoryBarrierWithGroupSync();
    }

    // The first thread writes the final value.
    if(local_thread_id == 0)
        output_tex[group_id.xy] = shared_mem[0];
}
