
// Meant to be used with no vertex buffer, 6 indices.
// Assumes counter-clockwise order, you're free to edit
// the shader if this isn't the case for you.

float4 main(uint vert_id : SV_VERTEXID) : SV_POSITION
{
    static const float4 verts[6] = {
        float4(-1.0f,  1.0f, 0.0f, 1.0f),  // Bottom-left tri
        float4(-1.0f, -1.0f, 0.0f, 1.0f),
        float4( 1.0f, -1.0f, 0.0f, 1.0f),
        float4(-1.0f,  1.0f, 0.0f, 1.0f),  // Top-right tri
        float4( 1.0f, -1.0f, 0.0f, 1.0f),
        float4( 1.0f,  1.0f, 0.0f, 1.0f),
    };

    return verts[vert_id];
}
