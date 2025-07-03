
Texture2D tex : register(t0, space2);
SamplerState s : register(s0, space2);

float4 main(float4 clip_pos : SV_POSITION) : SV_TARGET
{
    float2 uv = float2(clip_pos.x * 0.5f + 0.5f, 1.0f - (clip_pos.y * 0.5f + 0.5f));
    return tex.Sample(s, uv);
}
