
// Meant to be used with no vertex buffer, 6 indices.
// Assumes counter-clockwise order, you're free to edit
// the shader if this isn't the case for you.

struct Output
{
    float4 clip_pos : SV_POSITION;
    float2 uv : TEXCOORD0;
};

Output main(uint vert_id : SV_VERTEXID)
{
    static const float4 verts[6] = {
        float4(-1.0f,  1.0f, 0.0f, 1.0f),  // Bottom-left tri
        float4(-1.0f, -1.0f, 0.0f, 1.0f),
        float4( 1.0f, -1.0f, 0.0f, 1.0f),
        float4(-1.0f,  1.0f, 0.0f, 1.0f),  // Top-right tri
        float4( 1.0f, -1.0f, 0.0f, 1.0f),
        float4( 1.0f,  1.0f, 0.0f, 1.0f),
    };

    Output output;
    output.clip_pos = verts[vert_id];
    output.uv = float2(verts[vert_id].x * 0.5f + 0.5f, 1.0f - (verts[vert_id].y * 0.5f + 0.5f));
    return output;
}
