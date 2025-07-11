
cbuffer Uniforms : register(b0, space1)
{
    float4x4 world_to_view;
    float4x4 view_to_proj;
}

struct Output
{
    float4 clip_pos : SV_Position;
    float3 uv : TEXCOORD0;
};

float4x4 remove_translation(float4x4 mat)
{
    mat._m03 = 0.0f;
    mat._m13 = 0.0f;
    mat._m23 = 0.0f;
    return mat;
}

Output main(float3 pos : TEXCOORD0)
{
    float3 pos_norm = normalize(pos);
    float4x4 world_to_view_no_pos = remove_translation(world_to_view);
    float4 view_pos = mul(world_to_view_no_pos, float4(pos_norm, 1.0f));
    float4 clip_pos = mul(view_to_proj, view_pos);
    // Give infinite depth, so that no objects can go behind.
    clip_pos.z = clip_pos.w;

    Output output;
    output.clip_pos = clip_pos;
    output.uv       = float3(pos_norm);
    return output;
}
