
Texture2D tex : register(t0, space2);
SamplerState s : register(s0, space2);

float4 main(float2 uv : TEXCOORD0, float4 clip_pos : SV_POSITION) : SV_TARGET
{
    return tex.Sample(s, uv);
}
