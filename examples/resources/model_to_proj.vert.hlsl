
cbuffer Uniforms : register(b0, space1)
{
    float4x4 model_to_world;
    float4x4 model_to_world_normal;
    float4x4 world_to_proj;
}

struct Input
{
    float3 pos : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 lightmap_uv : TEXCOORD2;
};

struct Output
{
    float4 clip_pos : SV_POSITION;
    float3 world_pos : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 lightmap_uv : TEXCOORD2;
};

Output main(Input input)
{
    float4 world_pos = mul(model_to_world, float4(input.pos, 1.0f));
    float4 clip_pos  = mul(world_to_proj, world_pos);

    float4 world_normal = mul(model_to_world_normal, float4(input.normal, 1.0f));

    Output output;
    output.clip_pos = clip_pos;
    output.world_pos = (float3)world_pos;
    output.world_normal = (float3)world_normal;
    output.lightmap_uv = input.lightmap_uv;
    return output;
}
