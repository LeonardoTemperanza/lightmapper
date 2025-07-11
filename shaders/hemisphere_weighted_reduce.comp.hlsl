
// NOTE: This is similar to the other reduce shader, though there are differences.
// Here we first multiply the samples with a weight texture. (For hemicube-hemisphere
// conversion and custom material properties).
// We also perform a 3x1 reduction. This is because, due to the way the hemisphere
// batch texture is constructed, it is 3 * power-of-two horizontally, and power-of-two
// vertically. Since we want a single pixel for each hemisphere, we want to reduce
// horizontally first.
// It is a little bit of duplicated code but it makes the library
// easier to use (no need to set up #include path) which I think is worth it.

// Mostly from: https://therealmjp.github.io/posts/average-luminance-compute-shader/

static const uint THREAD_GROUP_SIZE = 8;
static const uint NUM_THREADS_IN_GROUP = THREAD_GROUP_SIZE * THREAD_GROUP_SIZE;
groupshared float4 shared_mem[NUM_THREADS_IN_GROUP];

Texture2D<float4> input_tex : register(t0, space0);
Texture2D<float2> weight_tex : register(t1, space0);
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
    uint2 weight_size;
    weight_tex.GetDimensions(weight_size.x, weight_size.y);

    const uint2 num_groups = output_size;
    //const uint2 num_samples_per_thread = uint2(3, 1);
    const uint2 num_used_threads_per_group = input_size / num_groups * uint2(3, 1);

    const uint local_thread_id = group_thread_id.y * THREAD_GROUP_SIZE + group_thread_id.x;

    const uint2 sample_idx = num_used_threads_per_group * group_id.xy + group_thread_id.xy * uint2(3, 1);
    const uint2 weight_sample_idx = sample_idx % weight_size;

    float4 s = (float4)0.0f;
    if(all(group_thread_id.xy < num_used_threads_per_group))
    {
        float4 s0 = input_tex[sample_idx + uint2(0, 0)];
        float4 s1 = input_tex[sample_idx + uint2(1, 0)];
        float4 s2 = input_tex[sample_idx + uint2(2, 0)];
        float2 w0 = weight_tex[weight_sample_idx + uint2(0, 0)];
        float2 w1 = weight_tex[weight_sample_idx + uint2(1, 0)];
        float2 w2 = weight_tex[weight_sample_idx + uint2(2, 0)];
        s = float4(s0.rgb * w0.r, s0.a * w0.g) +
            float4(s1.rgb * w1.r, s1.a * w1.g) +
            float4(s2.rgb * w2.r, s2.a * w2.g);
    }

    // Store sample in shared memory.
    shared_mem[local_thread_id] = s;
    GroupMemoryBarrierWithGroupSync();

    // Parallel reduction.
    [unroll(NUM_THREADS_IN_GROUP)]
    for(uint i = NUM_THREADS_IN_GROUP / 2; i > 0; i >>= 1)
    {
        if(local_thread_id < i)
            shared_mem[local_thread_id] += shared_mem[local_thread_id + i];

        GroupMemoryBarrierWithGroupSync();
    }

    // The first thread writes the final value.
    if(local_thread_id == 0)
        output_tex[group_id.xy] = shared_mem[0];
}
