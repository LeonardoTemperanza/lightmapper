
Texture2D<float4> source : register(t0, space2);

float4 main(float4 clip_pos : SV_POSITION) : SV_TARGET
{
    uint width, height;
    source.GetDimensions(width, height);

    float2 uv = float2(clip_pos.x, 1.0f - clip_pos.y) * 0.5f + 0.5f;
    return source.Load(int3(int2(uv * float2(width, height)), 0));
}
