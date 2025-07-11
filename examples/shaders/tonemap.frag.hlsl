
static const float exposure = -1.0f;

float3 linear_to_srgb(float3 color)
{
    bool3 cutoff = color.rgb < (float3)0.0031308;
    float3 higher = (float3)1.055 * pow(color.rgb, (float3)(1.0/2.4)) - (float3)0.055;
    float3 lower = color.rgb * (float3)12.92;

    return lerp(higher, lower, cutoff);
}

float3 tonemap_aces(float3 color)
{
    // ACES filmic tonemapper with highlight desaturation ("crosstalk").
    // Based on the curve fit by Krzysztof Narkowicz.
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/

    const float slope = 12.0f; // higher values = slower rise.

    // Store grayscale as an extra channel.
    float4 x = float4(
        // RGB
        color.r, color.g, color.b,
        // Luminosity
        (color.r * 0.299) + (color.g * 0.587) + (color.b * 0.114)
    );

    // ACES Tonemapper
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    float4 tonemap = clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
    float t = x.a;

    t = t * t / (slope + t);

    // Return after desaturation step.
    return lerp(tonemap.rgb, tonemap.aaa, t);
}

float3 hdr_to_ldr(float3 color)
{
    color *= pow(2.0f, exposure);
    return linear_to_srgb(tonemap_aces(color));
}

Texture2D tex : register(t0, space2);
SamplerState tex_sampler : register(s0, space2);

float4 main(float4 clip_pos : SV_POSITION, float2 uv : TEXCOORD0) : SV_TARGET
{
    return float4(hdr_to_ldr(linear_to_srgb(tex.Sample(tex_sampler, uv).rgb)), 1.0f);
}
