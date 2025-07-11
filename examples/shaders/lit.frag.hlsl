
Texture2D lightmap : register(t0, space2);
SamplerState lightmap_sampler : register(s0, space2);

struct Input
{
    float4 clip_pos : SV_POSITION;
    float3 world_pos    : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 lightmap_uv  : TEXCOORD2;
};

float4 main(Input input, bool is_front_facing : SV_IsFrontFace) : SV_TARGET
{
    // NOTE: For closed meshes, it's possible to disable backface culling and render additional
    // info into the alpha channel. This can improve the quality of the lightmap, so we do it here.
    return float4(lightmap.Sample(lightmap_sampler, input.lightmap_uv).rgb, is_front_facing ? 1.0 : 0.0);
}
