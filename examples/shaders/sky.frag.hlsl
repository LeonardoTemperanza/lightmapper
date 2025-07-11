
Texture2D sky_tex : register(t0, space2);
SamplerState sky_sampler : register(s0, space2);

static const float PI = 3.14159265f;

struct Input
{
    float4 clip_pos : SV_POSITION;
    float3 uv  : TEXCOORD0;
};

float3 sample_environment(float3 dir)
{
    float2 coords = float2((atan2(dir.x, dir.z) + PI) / (2*PI), acos(dir.y) / PI);
    return sky_tex.Sample(sky_sampler, coords).rgb;
}

float4 main(Input input) : SV_TARGET
{
    return float4(sample_environment(normalize(input.uv)), 1.0f);
}
