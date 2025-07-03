
Texture2D<float4> hemispheres : register(t0, space2);
Texture2D<float2> weights : register(t1, space2);

float4 weighted_sample(int2 hemi_uv, int2 weight_uv, int2 quadrant)
{
    float4 hemi_sample = hemispheres.Load(int3(hemi_uv, 0));
    float2 weight = weights.Load(int3(weight_uv + quadrant, 0));
    return float4(hemi_sample.rgb * weight.r, hemi_sample.a * weight.g);
}

float4 three_weighted_samples(int2 hemi_uv, int2 weight_uv, int2 offset)
{
    // Horizontal triple sum
    float4 sum = 0.0f;
    sum += weighted_sample(hemi_uv, weight_uv, offset + int2(0, 0));
    sum += weighted_sample(hemi_uv, weight_uv, offset + int2(2, 0));
    sum += weighted_sample(hemi_uv, weight_uv, offset + int2(4, 0));
    return sum;
}

// NOTE: Alpha contains the weighted count of valid samples
float4 main(float4 clip_pos : SV_POSITION) : SV_TARGET
{
    uint width, height;
    weights.GetDimensions(width, height);
    uint2 weights_texture_size = uint2(width, height);

    float2 in_uv = clip_pos.xy * float2(6.0f, 2.0f) + 0.5f;
    int2 hemi_uv   = int2(in_uv);
    int2 weight_uv = hemi_uv % weights_texture_size;

    float4 lb = three_weighted_samples(hemi_uv, weight_uv, int2(0, 0));
    float4 rb = three_weighted_samples(hemi_uv, weight_uv, int2(1, 0));
    float4 lt = three_weighted_samples(hemi_uv, weight_uv, int2(0, 1));
    float4 rt = three_weighted_samples(hemi_uv, weight_uv, int2(1, 1));
    return lb + rb + lt + rt;
}
