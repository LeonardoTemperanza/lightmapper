
// This is to fetch less attributes if only the depth is needed.

cbuffer Uniforms : register(b0, space1)
{
    float4x4 model_to_world;
    float4x4 model_to_world_normal;
    float4x4 world_to_proj;
}

float4 main(float3 pos : TEXCOORD0) : SV_POSITION
{
    float4 world_pos = mul(model_to_world, float4(pos, 1.0f));
    float4 clip_pos  = mul(world_to_proj, world_pos);
    return clip_pos;
}
