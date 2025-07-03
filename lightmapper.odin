
/*
MIT License

Copyright (c) 2025 Leonardo Temperanza

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

package lightmapper

import "core:log"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:math/linalg/glsl"
import la "core:math/linalg"
import intr "base:intrinsics"

import sdl "vendor:sdl3"

// General structure and algorithms from: https://github.com/ands/lightmapper

// The format is fixed to .R16G16B16A16_FLOAT. You're free
// to convert it to whichever format you prefer after the fact.
LIGHTMAP_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT

// The shaders are compiled by the user. This is because
// there are many different ways and formats to compile shaders in SDL_GPU.
// The shaders themselves are written in HLSL and can be found in this repository.
// You can then use SDL_ShaderCross, DXC or SPIRV_Cross to compile it or transpile it
// to a different format.
// (I treat compute pipelines like shaders here because they're easy to set up
// and they don't require any exterior knowledge like graphics pipelines do.)
// NOTE: Shaders must be released by the user.
Shaders :: struct
{
    hemi_reduce:          ^sdl.GPUComputePipeline,
    hemi_weighted_reduce: ^sdl.GPUComputePipeline,
    fullscreen_quad:      ^sdl.GPUShader,
    // Can't be a compute shader because textures can't
    // be a writable storage texture and sampleable at
    // the same time.
    blit_lightmap:        ^sdl.GPUShader,

    // Below are optional, only needed if wanting to call
    // their corresponding function.
    dilate: ^sdl.GPUComputePipeline,
    smooth: ^sdl.GPUComputePipeline,
    power:  ^sdl.GPUComputePipeline,
}

destroy_shaders :: proc(device: ^sdl.GPUDevice, shaders: Shaders)
{
    if shaders.hemi_reduce != nil do           sdl.ReleaseGPUComputePipeline(device, shaders.hemi_reduce)
    if shaders.hemi_weighted_reduce != nil do  sdl.ReleaseGPUComputePipeline(device, shaders.hemi_weighted_reduce)
    if shaders.fullscreen_quad != nil do       sdl.ReleaseGPUShader(device, shaders.fullscreen_quad)
    if shaders.blit_lightmap != nil do         sdl.ReleaseGPUShader(device, shaders.blit_lightmap)
    if shaders.dilate != nil do                sdl.ReleaseGPUComputePipeline(device, shaders.dilate)
    if shaders.smooth != nil do                sdl.ReleaseGPUComputePipeline(device, shaders.smooth)
    if shaders.power != nil do                 sdl.ReleaseGPUComputePipeline(device, shaders.power)
}

// Creates the context for this library. Can be used to render multiple lightmaps.
// You can create a new lightmap with make_lightmap and you can change the current lightmap
// with set_target_lightmap.
@(require_results)
init :: proc(device: ^sdl.GPUDevice,
             shaders: Shaders,
             hemisphere_target_format: sdl.GPUTextureFormat,        // Format of the hemisphere renderings, preferably HDR and with alpha.
             hemisphere_target_depth_format: sdl.GPUTextureFormat,  // Format of the depth of hemisphere renderings.
             hemisphere_resolution: int = 256,                      // Resolution of the hemisphere renderings.
             z_near: f32 = 0.01, z_far: f32 = 100,                  // Hemisphere min/max draw distance.
             background_color: [3]f32 = {},
             interpolation_passes: int = 4,                         // Hierarchical selective interpolation passes.
             interpolation_threshold: f32 = 0.01,                   // Error value below which lightmap pixels are interpolated instead of rendered.
             log_stats: bool = false,
             camera_to_surface_distance_modifier: f32 = 0.0         // Modifier for the height of the rendered hemispheres above the surface.
                                                                    // -1 -> stick to surface, 0 -> minimum height for interpolated surface normals,
                                                                    // > 0 -> improves gradients on surfaces with interpolated normals due to the flat surface horizon,
                                                                    // but may introduce other artifacts.
             ) -> Context
{
    // Validate args
    assert(hemisphere_resolution == 512 || hemisphere_resolution == 256 || hemisphere_resolution == 128 ||
           hemisphere_resolution == 64  || hemisphere_resolution == 32  || hemisphere_resolution == 16,
           "hemisphere_resolution must be a power of 2 and in the [16, 512] range.")
    assert(z_near < z_far, "z_near must be < z_far.")
    assert(z_near > 0.0, "z_near must be positive.")
    assert(camera_to_surface_distance_modifier >= -1.0, "camera_to_surface_distance_modifier must be >= -1.0.")
    assert(interpolation_passes >= 0 && interpolation_passes <= 8, "interpolation_passes must be in the [0, 8] range.")
    assert(interpolation_threshold >= 0.0, "interpolation_threshold must be >= 0.")
    validate_shaders(shaders)

    ctx: Context
    ctx.device = device
    ctx.cmd_buf = sdl.AcquireGPUCommandBuffer(ctx.device)
    ctx.shaders = shaders
    ctx.validation.ctx_initialized = true
    using ctx

    num_passes = 1 + 3 * u32(interpolation_passes)
    hemi_params.size = hemisphere_resolution
    hemi_params.z_near = z_near
    hemi_params.z_far  = z_far
    hemi_params.cam_to_surface_distance_modifier = camera_to_surface_distance_modifier
    hemi_params.clear_color = background_color
    hemi_params.batch_count = HEMI_BATCH_TEXTURE_SIZE / (hemi_params.size * [2]int{ 3, 1 })
    hemi_batch_to_lightmap = make([dynamic][2]int, hemi_params.batch_count.y * hemi_params.batch_count.x, hemi_params.batch_count.y * hemi_params.batch_count.x)

    // Build GPU Resources

    // Build pipelines
    blit_lightmap_pipeline = sdl.CreateGPUGraphicsPipeline(device, {
        target_info = sdl.GPUGraphicsPipelineTargetInfo {
            num_color_targets = 1,
            color_target_descriptions = raw_data([]sdl.GPUColorTargetDescription {
                { format = LIGHTMAP_FORMAT },
            }),
        },
        //vertex_input_state = static_mesh_layout,
        primitive_type = .TRIANGLELIST,
        rasterizer_state = {
            fill_mode = .FILL,
            cull_mode = .BACK,
            front_face = .COUNTER_CLOCKWISE,
        },
        multisample_state = {},
        depth_stencil_state = {
            enable_depth_test = false,
        },
        vertex_shader = shaders.fullscreen_quad,
        fragment_shader = shaders.blit_lightmap,
    })

    // Build textures
    default_weight_func :: proc(cos_theta: f32, user_data: rawptr) -> f32 { return 1.0 }
    set_hemisphere_weights(&ctx, default_weight_func, nil)

    target_usages := sdl.GPUTextureUsageFlags { .COLOR_TARGET, .COMPUTE_STORAGE_READ }
    ensure(sdl.GPUTextureSupportsFormat(device, hemisphere_target_format, .D2, target_usages), "The target format you're currently using is not supported for Lightmapper's usages.")
    hemi_batch_texture = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = hemisphere_target_format,
        width = auto_cast HEMI_BATCH_TEXTURE_SIZE.x,
        height = auto_cast HEMI_BATCH_TEXTURE_SIZE.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = target_usages,
    })

    depth_usages := sdl.GPUTextureUsageFlags { .DEPTH_STENCIL_TARGET }
    ensure(sdl.GPUTextureSupportsFormat(device, hemisphere_target_depth_format, .D2, depth_usages), "The depth format you're currently using is not supported for Lightmapper's usages.")
    hemi_batch_depth_texture = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = hemisphere_target_depth_format,
        width = auto_cast HEMI_BATCH_TEXTURE_SIZE.x,
        height = auto_cast HEMI_BATCH_TEXTURE_SIZE.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = depth_usages,
    })

    reduced_usages := sdl.GPUTextureUsageFlags { .COMPUTE_STORAGE_READ, .COMPUTE_STORAGE_WRITE }
    ensure(sdl.GPUTextureSupportsFormat(device, .R32G32B32A32_FLOAT, .D2, reduced_usages))
    for i in 0..<2
    {
        hemi_reduce_textures[i] = sdl.CreateGPUTexture(device, {
            type = .D2,
            format = .R32G32B32A32_FLOAT,
            width = auto_cast HEMI_BATCH_TEXTURE_SIZE.x,
            height = auto_cast HEMI_BATCH_TEXTURE_SIZE.y,
            layer_count_or_depth = 1,
            num_levels = 1,
            usage = reduced_usages,
        })
    }

    lm_tmp_usages := sdl.GPUTextureUsageFlags { .COMPUTE_STORAGE_READ }
    ensure(sdl.GPUTextureSupportsFormat(device, .R32G32B32A32_FLOAT, .D2, lm_tmp_usages))
    lightmap_tmp_texture = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = .R32G32B32A32_FLOAT,
        width = auto_cast HEMI_BATCH_TEXTURE_SIZE.x,
        height = auto_cast HEMI_BATCH_TEXTURE_SIZE.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = reduced_usages,
    })

    return ctx
}

// Must be called after you're done.
destroy :: proc(using ctx: ^Context)
{
    assert(validation.ctx_initialized, "Attempting to destroy a Lightmapper context without having initialized it first!")
    assert(!validation.iter_begin, "Forgot to call bake_iterate_end! It must be called after each call to bake_iterate_begin (iff it returns true).")

    // CPU resources.
    delete(hemi_batch_to_lightmap)

    // Destroy textures.
    sdl.ReleaseGPUTexture(device, weights_texture)
    sdl.ReleaseGPUTexture(device, hemi_batch_texture)
    sdl.ReleaseGPUTexture(device, hemi_batch_depth_texture)
    sdl.ReleaseGPUTexture(device, hemi_reduce_textures[0])
    sdl.ReleaseGPUTexture(device, hemi_reduce_textures[1])
    sdl.ReleaseGPUTexture(device, lightmap_tmp_texture)

    // Destroy pipelines.
    sdl.ReleaseGPUGraphicsPipeline(device, blit_lightmap_pipeline)
}

Mesh :: struct
{
    positions: Buffer,  // Expected to be a vector3.
    normals: Buffer,    // Expected to be a vector3.
    lm_uvs: Buffer,     // Expected to be a vector2.
    indices: Buffer,    // Expected to be a scalar value.
    tri_count: int,
    use_indices: bool,
}

// "OpenGL style" of specifying buffer format.
Buffer :: struct
{
    data: rawptr,
    type: Type,
    stride: int,
    offset: uintptr,
}

Type :: enum
{
    None,
    U8,
    U16,
    U32,
    S8,
    S16,
    S32,
    F32,
}

set_target_mesh :: proc(ctx: ^Context, mesh: Mesh, model_to_world: matrix[4, 4]f32)
{
    ctx.mesh = mesh
    ctx.mesh_transform = model_to_world
    ctx.mesh_normal_mat = la.transpose(la.inverse(model_to_world))

    set_cursor_and_rasterizer(ctx, 0)
}

Lightmap :: struct
{
    tex:  ^sdl.GPUTexture,
    size: [2]int,
}

// This texture is sampleable so it can be fed back to your
// renderer, to get multiple bounces.
// NOTE: Lightmaps must be disposed of by the user.
@(require_results)
make_lightmap :: proc(using ctx: ^Context, size: [2]int) -> Lightmap
{
    res: Lightmap
    res.tex = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = LIGHTMAP_FORMAT,
        width = auto_cast size.x,
        height = auto_cast size.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = { .SAMPLER, .COLOR_TARGET },
    })
    res.size = size
    return res
}

delete_lightmap :: proc(ctx: ^Context, lightmap: Lightmap)
{
    sdl.ReleaseGPUTexture(ctx.device, lightmap.tex)
}

// One texture can hold lightmaps for multiple meshes.
set_target_lightmap :: proc(ctx: ^Context, lightmap: Lightmap)
{
    ctx.lightmap = lightmap
}

// Optional: Set material characteristics by specifying cos(theta)-dependent weights for incoming light.
// NOTE: This is expensive as this builds and uploads a texture so preferably use it on startup.
Weight_Func_Type :: proc(cos_theta: f32, userdata: rawptr)->f32
set_hemisphere_weights :: proc(using ctx: ^Context,
                               weight_func: Weight_Func_Type,
                               user_data: rawptr,
                               allocator := context.allocator)
{
    if weights_texture != nil {
        sdl.ReleaseGPUTexture(device, weights_texture)
    }

    hemi_size := hemi_params.size
    weights := make([]f32, hemi_size * hemi_size * 2 * 3, allocator = allocator)
    defer delete(weights)

    center  := f32(hemi_size - 1) * 0.5
    sum     := 0.0
    for y in 0..<hemi_size
    {
        dy := 2.0 * (f32(y) - center) / f32(hemi_size)
        for x in 0..<hemi_size
        {
            dx := 2.0 * (f32(x) - center) / f32(hemi_size)
            v := normalize([3]f32 { dx, dy, 1.0 })
            solid_angle := v.z * v.z * v.z

            w0 := weights[2 * (y * (3 * hemi_size) + x):]
            w1 := w0[2 * hemi_size:]
            w2 := w1[2 * hemi_size:]

            // Center weights.
            w0[0] = solid_angle * weight_func(v.z, user_data)
            w0[1] = solid_angle

            // Left/Right side weights.
            w1[0] = solid_angle * weight_func(abs(v.x), user_data)
            w1[1] = solid_angle

            // Up/Down side weights.
            w2[0] = solid_angle * weight_func(abs(v.y), user_data)
            w2[1] = solid_angle

            sum += 3.0 * f64(solid_angle)
        }
    }

    // Normalize weights.
    weights_scale := f32(1.0 / sum)  // (Faster to multiply than to divide)
    for &w in weights {
        w *= weights_scale
    }

    // Upload to GPU.
    usage  := sdl.GPUTextureUsageFlags { .SAMPLER }
    type   := sdl.GPUTextureType.D2
    format := sdl.GPUTextureFormat.R32G32_FLOAT
    width  := u32(3 * hemi_size)
    height := u32(hemi_size)
    ensure(sdl.GPUTextureSupportsFormat(device, format, type, usage))
    weights_texture = sdl.CreateGPUTexture(device, {
        type = type,
        format = format,
        width = width,
        height = height,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = usage
    })

    transfer_buf := sdl.CreateGPUTransferBuffer(device, {
        usage = .UPLOAD,
        size  = auto_cast len(weights) * size_of(f32)
    })
    defer sdl.ReleaseGPUTransferBuffer(device, transfer_buf)

    transfer_dst := sdl.MapGPUTransferBuffer(device, transfer_buf, false)
    intr.mem_copy(transfer_dst, raw_data(weights), len(weights))
    sdl.UnmapGPUTransferBuffer(device, transfer_buf)

    cmd_buf_ := sdl.AcquireGPUCommandBuffer(device)
    pass := sdl.BeginGPUCopyPass(cmd_buf_)
    sdl.UploadToGPUTexture(
        pass,
        source = {
            transfer_buffer = transfer_buf,
            offset = 0,
        },
        destination = {
            texture = weights_texture,
            w = width,
            h = height,
            d = 1,
        },
        cycle = false
    )
    sdl.EndGPUCopyPass(pass)

    ok_s := sdl.SubmitGPUCommandBuffer(cmd_buf_)
    ensure(ok_s)
}

// This describes how your scene should be rendered
// for correct/optimal (depending on the parameter) behavior.
Scene_Render_Params :: struct
{
    depth_only: bool,        // Optional, this is to speed the first pass up.
    render_shadowmap: bool,  // Optional, if you want to include a directional light source.

    viewport_offset: [2]int,
    viewport_size:   [2]int,
    world_to_view: matrix[4, 4]f32,
    view_to_proj: matrix[4, 4]f32,

    pass: ^sdl.GPURenderPass,
    cmd_buf: ^sdl.GPUCommandBuffer,
}

// Can be used like this:
// for render_params in lm.bake_iterate_begin(lm_ctx)
bake_iterate_begin :: proc(using ctx: ^Context) -> (params: Scene_Render_Params, proceed: bool)
{
    assert(!validation.iter_begin, "Forgot to call bake_iterate_end! It must be called after each call to bake_iterate_begin (iff it returns true).")
    validation.iter_begin = true
    validate_context(ctx)

    // If bake render pass has been ended, restart it.
    if bake_pass == nil
    {
        // Start new render pass
        // Render to the hemisphere batch texture.
        color_target := sdl.GPUColorTargetInfo {
            texture = hemi_batch_texture,
            clear_color = { hemi_params.clear_color.x, hemi_params.clear_color.y, hemi_params.clear_color.z, 1.0 },
            load_op = .CLEAR,
            store_op = .STORE,
        }
        depth_target := sdl.GPUDepthStencilTargetInfo {
            texture = hemi_batch_depth_texture,
            clear_depth = 1.0,
            load_op = .CLEAR,
            store_op = .STORE,
            stencil_load_op = .DONT_CARE,
            stencil_store_op = .DONT_CARE,
        }
        bake_pass = sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, &depth_target)
    }

    for cursor.hemi_side >= 5
    {
        move_to_next_potential_rasterizer_position(ctx)
        found := find_first_rasterizer_position(ctx)
        if found
        {
            cursor.hemi_side = 0
            break
        }

        // There are no valid sample positions on the current triangle,
        // so try to move onto the next triangle.

        triangles_left := cursor.tri_base_idx + 3 < mesh.tri_count
        if triangles_left
        {
            // Move onto the next triangle.
            set_cursor_and_rasterizer(ctx, auto_cast cursor.tri_base_idx + 3)
        }
        else
        {
            cursor.pass += 1
            if cursor.pass < 4 //num_passes
            {
                // Start over with a new pass
                set_cursor_and_rasterizer(ctx, 0)
            }
            else
            {
                // We've finished the lightmapping process.
                if bake_pass != nil
                {
                    sdl.EndGPURenderPass(bake_pass)
                    bake_pass = nil
                }

                // Integrate and store last batch.
                integrate_hemisphere_batch_and_copy_to_storage_tex(ctx)

                // Write to the final image.
                {
                    // Setup and start a render pass
                    target_info := sdl.GPUColorTargetInfo {
                        texture = lightmap.tex,
                        clear_color = {},
                        load_op = .CLEAR,
                        store_op = .STORE,
                        mip_level = 0,
                        layer_or_depth_plane = 0
                    }
                    pass := sdl.BeginGPURenderPass(cmd_buf, &target_info, 1, nil)
                    defer sdl.EndGPURenderPass(pass)

                    sdl.BindGPUGraphicsPipeline(pass, blit_lightmap_pipeline)
                    sdl.BindGPUFragmentStorageTextures(pass, 0, &lightmap_tmp_texture, 1)
                    sdl.DrawGPUPrimitives(
                        pass,
                        num_vertices   = 6,
                        num_instances  = 1,
                        first_vertex   = 0,
                        first_instance = 0
                    )
                }

                // Submit command buffer.
                ok := sdl.SubmitGPUCommandBuffer(cmd_buf)
                assert(ok)

                // Reset baking state.
                cursor.pass = 0
                cursor.hemi_side = 5

                // Don't call bake_iterate_end anymore.
                validation.iter_begin = false
                return {}, false
            }
        }
    }

    viewport_offset, viewport_size, world_to_view, view_to_proj := compute_current_camera(ctx)
    params.viewport_offset = viewport_offset
    params.viewport_size   = viewport_size
    params.world_to_view   = world_to_view
    params.view_to_proj    = view_to_proj
    params.pass = bake_pass
    params.cmd_buf = cmd_buf
    assert(bake_pass != nil)
    assert(cmd_buf != nil)
    return params, true
}

// Returns a value from 0 to 1.
bake_progress :: proc(using ctx: ^Context) -> f32
{
    pass_progress := f32(cursor.tri_base_idx) / (f32(mesh.tri_count) * 3.0)
    return (f32(cursor.pass) + pass_progress) / f32(num_passes)
}

// Must be called if and only if bake_iterate_begin returns true. For example:
// for render_params in lm.bake_iterate_begin(lm_ctx)
// {
//     defer lm.bake_iterate_end(lm_ctx)
//     ...
// }
bake_iterate_end :: proc(using ctx: ^Context)
{
    assert(validation.iter_begin, "bake_iterate_end should only be called after bake_iterate_begin and iff that returns true!")
    validation.iter_begin = false
    validate_context(ctx)

    cursor.hemi_side += 1
    if cursor.hemi_side >= 5
    {
        cursor.hemi_idx += 1
        was_last_in_batch := cursor.hemi_idx >= hemi_params.batch_count.x * hemi_params.batch_count.y
        if was_last_in_batch
        {
            if bake_pass != nil
            {
                sdl.EndGPURenderPass(bake_pass)
                bake_pass = nil
            }

            integrate_hemisphere_batch_and_copy_to_storage_tex(ctx)
            cursor.hemi_idx = 0
        }
        else
        {
            hemi_batch_to_lightmap[cursor.hemi_idx] = rasterizer.pos
        }
    }
}

////////////////////////////
// Internal

Context :: struct
{
    // Settings
    num_passes: u32,
    interp_threshold: f32,
    lightmap: Lightmap,
    lightmap_region_size: [2]int,
    lightmap_region_offset: [2]int,
    do_log_stats: bool,

    // Validation
    validation: Validation,

    // Bound state
    mesh: Mesh,
    mesh_transform:  matrix[4, 4]f32,
    mesh_normal_mat: matrix[4, 4]f32,

    // Lightmap baking state
    cursor: Cursor,
    rasterizer: Rasterizer,
    tri_sample: Tri_Sample,
    hemi_params: Hemisphere_Params,
    hemi_batch_to_lightmap: [dynamic][2]int,  // hemisphere idx in batch -> lightmap pixel

    // GPU Resources
    device: ^sdl.GPUDevice,
    cmd_buf: ^sdl.GPUCommandBuffer,
    bake_pass: ^sdl.GPURenderPass,
    shaders: Shaders,

    // Pipelines
    blit_lightmap_pipeline: ^sdl.GPUGraphicsPipeline,

    // Textures
    weights_texture: ^sdl.GPUTexture,  // Holds the weights to convert pixels from hemicube to hemisphere.
    hemi_batch_texture: ^sdl.GPUTexture,  // Holds many hemispheres (which in turn hold 5 different hemicube sides)
    hemi_batch_depth_texture: ^sdl.GPUTexture,
    hemi_reduce_textures: [2]^sdl.GPUTexture,  // Ping-pong buffers used to run hemisphere reduction steps.
    // Used to temporarily store the lightmap values as 32-bit floats before
    // scaling down the values (based on validity)
    lightmap_tmp_texture: ^sdl.GPUTexture,
}

Validation :: struct
{
    ctx_initialized: bool,
    iter_begin: bool,
}

Cursor :: struct
{
    pass: u32,
    tri_base_idx: int,  // The index of the first vertex.
    tri_verts_pos: [3] [3]f32,
    tri_verts_normal: [3] [3]f32,
    tri_verts_lm_uv: [3] [2]f32,

    hemi_idx:  int,  // Index in the current batch.
    hemi_side: int,  // [0, 4], side of the hemicube.
}

Tri_Sample :: struct
{
    pos: [3]f32,
    dir: [3]f32,
    up:  [3]f32,
}

// This is a conservative rasterizer, meaning if any point
// of the triangle overlaps with the pixel it will get rasterized.
Rasterizer :: struct
{
    min: [2]int,
    max: [2]int,
    pos: [2]int,
}

HEMI_BATCH_TEXTURE_SIZE :: [2]int { 512 * 3, 512 }

Hemisphere_Params :: struct
{
    z_near: f32,
    z_far: f32,
    size: int,
    cam_to_surface_distance_modifier: f32,
    clear_color: [3]f32, // Do i actually need this?
    batch_count: [2]int,
}

compute_current_camera :: proc(using ctx: ^Context) -> (viewport_offset: [2]int, viewport_size: [2]int, world_to_view: matrix[4, 4]f32, view_to_proj: matrix[4, 4]f32)
{
    assert(cursor.hemi_side >= 0 && cursor.hemi_side < 5)

    x := (cursor.hemi_idx % hemi_params.batch_count.x) * hemi_params.size * 3
    y := (cursor.hemi_idx / hemi_params.batch_count.y) * hemi_params.size

    size := hemi_params.size
    zn := hemi_params.z_near
    zf := hemi_params.z_far

    pos := tri_sample.pos
    dir := tri_sample.dir
    up  := tri_sample.up
    right := cross(dir, up)

    // +-------+---+---+-------+
    // |       |   |   |   D   |
    // |   C   | R | L +-------+
    // |       |   |   |   U   |
    // +-------+---+---+-------+
    switch cursor.hemi_side
    {
        case 0:  // Center
        {
            world_to_view, view_to_proj = compute_matrices(pos, dir, up, -zn, zn, -zn, zn, zn, zf)
            viewport_offset = { x, y }
            viewport_size   = { size, size }
        }
        case 1:  // Right
        {
            world_to_view, view_to_proj = compute_matrices(pos, right, up, -zn, 0, -zn, zn, zn, zf)
            viewport_offset = { x + size, y }
            viewport_size   = { size / 2.0, size }
        }
        case 2:  // Left
        {
            world_to_view, view_to_proj = compute_matrices(pos, -right, up, 0, zn, -zn, zn, zn, zf)
            viewport_offset = { x + size + size / 2.0, y }
            viewport_size   = { size / 2.0, size }
        }
        case 3:  // Down
        {
            world_to_view, view_to_proj = compute_matrices(pos, -up, dir, -zn, zn, 0, zn, zn, zf)
            viewport_offset = { x + size + size, y + size / 2.0 }
            viewport_size   = { size, size / 2.0 }
        }
        case 4:  // Up
        {
            world_to_view, view_to_proj = compute_matrices(pos, up, -dir, -zn, zn, -zn, 0, zn, zf)
            viewport_offset = { x + size + size, y }
            viewport_size   = { size, size / 2.0 }
        }
    }

    return viewport_offset, viewport_size, world_to_view, view_to_proj

    compute_matrices :: proc(pos: [3]f32, dir: [3]f32, up: [3]f32,
                             l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) ->
                             (world_to_view: matrix[4, 4]f32, view_to_proj: matrix[4, 4]f32)
    {
        side := cross(dir, up)
        world_to_view = {
            side.x,          up.x,          -dir.x,          0.0,
            side.y,          up.y,          -dir.y,          0.0,
            side.z,          up.z,          -dir.z,          0.0,
            dot(side, -pos), dot(up, -pos), dot(-dir, -pos), 1.0,
        }

        // Orthographic view
        view_to_proj = matrix[4, 4]f32 {
            2.0 / (r - l),  0,                  0,              -(r + l) / (r - l),
            0,                  2.0 / (t - b),  0,              -(t + b) / (t - b),
            0,                  0,                  1.0 / (f - n),  -n / (f - n),
            0,                  0,                  0,              1,
        }

        return world_to_view, view_to_proj
    }
}

set_cursor_and_rasterizer :: proc(using ctx: ^Context, tri_idx: i64)
{
    cursor.tri_base_idx = auto_cast tri_idx

    verts_indices: [3]i64 = tri_idx + 0
    if mesh.use_indices {
        for i in 0..<3 do verts_indices[i] = get_i64_from_buffer(mesh.indices, tri_idx + auto_cast i)
    } else {
        for i in 0..<3 do verts_indices[i] = tri_idx + auto_cast i
    }

    uv_scale := [2]f32 { auto_cast lightmap.size.x, auto_cast lightmap.size.y }

    uv_min: [2]f32 = max(f32)
    uv_max: [2]f32 = min(f32)
    for i in 0..<3
    {
        cursor.tri_verts_pos[i]    = get_vec3f32_from_buffer(mesh.positions, verts_indices[i])
        cursor.tri_verts_normal[i] = get_vec3f32_from_buffer(mesh.normals, verts_indices[i])
        cursor.tri_verts_lm_uv[i]  = get_vec2f32_from_buffer(mesh.lm_uvs, verts_indices[i])

        // Transformations.
        pos := [4]f32 { cursor.tri_verts_pos[i].x, cursor.tri_verts_pos[i].y, cursor.tri_verts_pos[i].z, 1.0 }
        cursor.tri_verts_pos[i]    = (mesh_transform * pos).xyz
        normal := [4]f32 { cursor.tri_verts_normal[i].x, cursor.tri_verts_normal[i].y, cursor.tri_verts_normal[i].z, 1.0 }
        cursor.tri_verts_pos[i] = (mesh_normal_mat * normal).xyz

        cursor.tri_verts_lm_uv[i] *= uv_scale

        uv_min = glsl.min(uv_min, cursor.tri_verts_lm_uv[i])
        uv_max = glsl.max(uv_max, cursor.tri_verts_lm_uv[i])
    }

    // Calculate bounding box on lightmap for conservative rasterization.
    bb_min := la.floor(uv_min)
    bb_max := la.ceil(uv_max)
    rasterizer.min.x = min(int(bb_min.x), 0)
    rasterizer.min.y = min(int(bb_min.y), 0)
    rasterizer.max.x = min(int(bb_max.x) + 1, lightmap.size.x)
    rasterizer.max.y = min(int(bb_max.y) + 1, lightmap.size.y)
    assert(rasterizer.min.x <= rasterizer.max.x && rasterizer.min.y <= rasterizer.max.y)
    rasterizer.pos = rasterizer.min + pass_offset(ctx)

    // Check if there are any valid samples on this triangle.
    if (rasterizer.pos.x <= rasterizer.max.x && rasterizer.pos.y <= rasterizer.max.y &&
        find_first_rasterizer_position(ctx))
    {
        cursor.hemi_side = 0
    }
    else
    {
        cursor.hemi_side = 5  // Already finished rasterizing this triangle before even having begun.
    }
}

// NOTE: This starts new passes, so when calling this function no other pass should be bound.
integrate_hemisphere_batch_and_copy_to_storage_tex :: proc(using ctx: ^Context)
{
    zeros: [32]uint
    sdl.PushGPUComputeUniformData(cmd_buf, 0, &zeros, size_of(zeros))

    hemi_size_result := hemi_params.size
    thread_group_size := 8  // NOTE: Coupled to the shader code.
    num_samples_per_group := 2 * thread_group_size

    tex_target_idx := 0
    tex_read_idx   := 1

    // First pass: Weighted downsampling.
    {
        // Read from hemi_batch_texture and write to hemi_reduce_textures[0]
        storage_write_tex_binding := sdl.GPUStorageTextureReadWriteBinding {
            texture = hemi_reduce_textures[tex_target_idx],
            mip_level = 0,
            layer = 0,
            cycle = false
        }
        pass := sdl.BeginGPUComputePass(cmd_buf, &storage_write_tex_binding, 1, nil, 0)
        defer sdl.EndGPUComputePass(pass)

        sdl.BindGPUComputePipeline(pass, shaders.hemi_weighted_reduce)

        storage_tex_bindings := []^sdl.GPUTexture { hemi_batch_texture, weights_texture }
        sdl.BindGPUComputeStorageTextures(pass, 0, &storage_tex_bindings[0], 2)
        group_count: u32 = auto_cast (hemi_size_result / num_samples_per_group)
        assert(group_count > 0)
        sdl.DispatchGPUCompute(pass, group_count, group_count, 1)

        // Update size.
        hemi_size_result /= num_samples_per_group
    }

    // Successive passes: non-weighted downsampling passes.
    for hemi_size_result > 1
    {
        // Read from hemi_reduce_textures[read] and write to hemi_reduce_textures[target]
        storage_tex_binding := sdl.GPUStorageTextureReadWriteBinding {
            texture = hemi_reduce_textures[tex_target_idx],
            mip_level = 0,
            layer = 0,
            cycle = false
        }
        pass := sdl.BeginGPUComputePass(cmd_buf, &storage_tex_binding, 1, nil, 0)
        defer sdl.EndGPUComputePass(pass)

        sdl.BindGPUComputePipeline(pass, shaders.hemi_reduce)

        sdl.BindGPUComputeStorageTextures(pass, 0, &hemi_reduce_textures[tex_read_idx], 1)
        group_count: u32 = auto_cast (hemi_size_result / num_samples_per_group)
        assert(group_count > 0)
        sdl.DispatchGPUCompute(pass, group_count, group_count, 1)

        // Swap textures.
        tex_target_idx = tex_read_idx
        tex_read_idx = (tex_target_idx + 1) % 2

        // Update size.
        hemi_size_result /= num_samples_per_group
    }

    // Swap back textures to undo last iteration.
    tex_target_idx = tex_read_idx
    tex_read_idx = (tex_target_idx + 1) % 2

    // Copy results to storage texture.
    {
        pass := sdl.BeginGPUCopyPass(cmd_buf)
        defer sdl.EndGPUCopyPass(pass)

        src := sdl.GPUTextureLocation {
            texture = hemi_reduce_textures[tex_target_idx],
            mip_level = 0,
            layer = 0,
            x = 0,
            y = 0,
            z = 0
        }
        dst := sdl.GPUTextureLocation {
            texture = lightmap_tmp_texture,
            mip_level = 0,
            layer = 0,
            x = auto_cast hemi_batch_to_lightmap[0].x,
            y = auto_cast hemi_batch_to_lightmap[0].y,
            z = 0
        }
        sdl.CopyGPUTextureToTexture(pass, src, dst, 1, 1, 1, false)
    }



    // Update map.


    // Advance storage texture position.
}

find_first_rasterizer_position :: proc(using ctx: ^Context) -> bool
{
    for !try_sampling_rasterizer_position(ctx)
    {
        move_to_next_potential_rasterizer_position(ctx)
        if has_rasterizer_finished(ctx) {
            return false
        }
    }

    return true
}

move_to_next_potential_rasterizer_position :: proc(using ctx: ^Context)
{
    step := pass_step_size(ctx)
    rasterizer.pos.x += step
    for rasterizer.pos.x >= rasterizer.max.x
    {
        rasterizer.pos.x = rasterizer.min.x + pass_offset(ctx).x
        rasterizer.pos.y += step

        if has_rasterizer_finished(ctx) do break
    }
}

// If it returns true, ctx.tri_sample will be filled in.
try_sampling_rasterizer_position :: proc(using ctx: ^Context) -> bool
{
    if has_rasterizer_finished(ctx) do return false

    // TODO: Check if lightmap was already set?

    // Try computing centroid by clipping the pixel against the triangle.
    raster_pos := [2]f32 { auto_cast rasterizer.pos.x, auto_cast rasterizer.pos.y }
    clipped_array, clipped_len := aabb_tri_clip(raster_pos + 0.5, 1.0, cursor.tri_verts_lm_uv)
    if clipped_len <= 0 do return false  // Nothing left.

    clipped := clipped_array[:clipped_len]

    // Compute centroid position and area.
    // http://the-witness.net/news/2010/09/hemicube-rendering-and-integration/
    // Centroid sampling basically makes it so we don't clip inside of a wall
    // for hemisphere rendering of an intersecting floor.
    clipped_first := clipped[0]
    clipped_last  := clipped[len(clipped) - 1]
    centroid := clipped_first
    area := clipped_last.x * clipped_first.y - clipped_last.y * clipped_first.x
    for i in 1..<len(clipped)
    {
        centroid += clipped[i]
        area += clipped[i - 1].x * clipped[i].y - clipped[i - 1].y * clipped[i].x
    }
    centroid = centroid / auto_cast len(clipped)
    area /= 2.0

    if area <= 0.0 do return false  // No area left.

    // Compute barycentric coords.
    uv := to_barycentric(cursor.tri_verts_lm_uv[0], cursor.tri_verts_lm_uv[1], cursor.tri_verts_lm_uv[2], centroid)
    if math.is_nan(uv.x) || math.is_inf(uv.x) || math.is_nan(uv.y) || math.is_inf(uv.y) {
        return false // Degenerate case.
    }

    // Try to interpolate color from neighbors.
    /*
    if cursor.pass > 0
    {

    }
    */

    // Could not interpolate. Must render a hemisphere.
    // Compute 3D sample position and orientation.

    p0 := cursor.tri_verts_pos[0]
    p1 := cursor.tri_verts_pos[1]
    p2 := cursor.tri_verts_pos[2]
    v1 := p1 - p0
    v2 := p2 - p0
    tri_sample.pos = p0 + v2 * uv.x + v1 * uv.y

    n0 := cursor.tri_verts_normal[0]
    n1 := cursor.tri_verts_normal[1]
    n2 := cursor.tri_verts_normal[2]
    nv1 := n1 - n0
    nv2 := n2 - n0
    tri_sample.dir = normalize(n0 + nv2 * uv.x + nv1 * uv.y)
    camera_to_surface_distance := (1.0 + hemi_params.cam_to_surface_distance_modifier) * hemi_params.z_near * math.sqrt_f32(2.0)
    tri_sample.pos += tri_sample.dir * camera_to_surface_distance

    if is_inf(tri_sample.pos) || is_inf(tri_sample.dir) || la.length(tri_sample.dir) < 0.5 {
        return false
    }

    up := [3]f32 { 0, 1, 0 }
    if abs(dot(up, tri_sample.dir)) > 0.8 {
        up = [3]f32 { 0, 0, 1 }
    }

    // http://the-witness.net/news/2010/09/hemicube-rendering-and-integration/
    // Randomized directions fix banding artifacts, though they increase noise.
    // Noise is much easier to deal with, by smoothing the lightmap a bit.
    when false
    {
        tri_sample.up = normalize(cross(up, tri_sample.dir))
    }
    else
    {
        side := normalize(cross(up, tri_sample.dir))
        up = normalize(cross(side, tri_sample.dir))
        rx := rasterizer.pos.x % 3
        ry := rasterizer.pos.y % 3

        base_angle: f32 = 0.1
        base_angles := [3][3]f32 {
            { base_angle, base_angle + 1.0 / 3.0, base_angle + 2.0 / 3.0 },
            { base_angle + 1.0 / 3.0, base_angle + 2.0 / 3.0, base_angle },
            { base_angle + 2.0 / 3.0, base_angle, base_angle + 1.0 / 3.0 }
        }
        phi := 2.0 * math.PI * base_angles[ry][rx] + 0.1 * rand.float32()

        tri_sample.up = normalize(side * math.cos(phi) + up * math.sin(phi))
    }

    return true
}

aabb_tri_clip :: proc(aabb_center: [2]f32, aabb_size: [2]f32, tri: [3][2]f32) -> (res: [16][2]f32, num: int)
{
    aabb := [16][2]f32 {
        { aabb_center.x - aabb_size.x * 0.5, aabb_center.y + aabb_size.y * 0.5 },
        { aabb_center.x + aabb_size.x * 0.5, aabb_center.y + aabb_size.y * 0.5 },
        { aabb_center.x + aabb_size.x * 0.5, aabb_center.y - aabb_size.y * 0.5 },
        { aabb_center.x - aabb_size.x * 0.5, aabb_center.y - aabb_size.y * 0.5 },
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    }

    n_poly := 4
    num = n_poly
    dir := left_of(tri[0], tri[1], tri[2])
    for i in 0..<3
    {
        j := i - 1
        if i == 0 do j = 2
        if i != 0
        {
            for n_poly in 0..<num {
                aabb[n_poly] = res[n_poly]
            }
        }

        num = 0
        v0 := aabb[n_poly - 1]
        side_0 := left_of(tri[j], tri[i], v0)
        if side_0 != -dir
        {
            res[num] = v0
            num += 1
        }

        for k in 0..<n_poly
        {
            v1 := aabb[k]
            x: [2]f32
            side_1 := left_of(tri[j], tri[i], v1)
            intersect, inter_p := line_intersection(tri[j], tri[i], v0, v1)
            if side_0 + side_1 == 0 && side_0 != 0 && intersect {
                res[num] = x
                num += 1
            }

            if k == n_poly - 1 do break

            if side_1 != -dir
            {
                res[num] = v1
                num += 1
            }

            v0 = v1
            side_0 = side_1
        }
    }

    return res, num

    left_of :: proc(a: [2]f32, b: [2]f32, c: [2]f32) -> int
    {
        v0 := b - a
        v1 := c - b
        res := v0.x * v1.y - v0.y * v1.x
        if res < 0 do return -1
        if res > 0 do return 1
        return 0
    }

    line_intersection :: proc(x0: [2]f32, x1: [2]f32, y0: [2]f32, y1: [2]f32) -> (intersect: bool, p: [2]f32)
    {
        dx := x1 - x0
        dy := y1 - y0
        d := x0 - y0
        dyx := dy.x * dx.y - dy.y * dx.x
        if dyx == 0.0 do return false, {}

        dyx = (d.x * dx.y - d.y * dx.x) / dyx
        if dyx <= 0 || dyx >= 1 do return false, {}

        p = { y0.x + dyx * dy.x, y0.y + dyx * dy.y }
        return true, p
    }
}

// From: http://www.blackpawn.com/texts/pointinpoly/
to_barycentric :: proc(p1: [2]f32, p2: [2]f32, p3: [2]f32, p: [2]f32) -> [2]f32
{
    v0 := p3 - p1
    v1 := p2 - p1
    v2 := p - p1
    dot00 := dot(v0, v0)
    dot01 := dot(v0, v1)
    dot02 := dot(v0, v2)
    dot11 := dot(v1, v1)
    dot12 := dot(v1, v2)
    inv_denom := 1.0 / (dot00 * dot11 - dot01 * dot01)
    u := (dot11 * dot02 - dot01 * dot12) * inv_denom
    v := (dot00 * dot12 - dot01 * dot02) * inv_denom
    return { u, v }
}

has_rasterizer_finished :: proc(using ctx: ^Context) -> bool
{
    return rasterizer.pos.y >= rasterizer.max.y
}

// Pass order of one 4x4 interpolation patch for two interpolation steps (and the next neighbors right of/below it).
// 0 4 1 4 0
// 5 6 5 6 5
// 2 4 3 4 2
// 5 6 5 6 5
// 0 4 1 4 0

pass_step_size :: proc(using ctx: ^Context) -> int
{
    pass_minus_one := cursor.pass - 1 if cursor.pass > 0 else 0
    shift := num_passes / 3 - pass_minus_one / 3
    step  := 1 << shift
    assert(step > 0)
    return step
}

pass_offset :: proc(using ctx: ^Context) -> [2]int
{
    if cursor.pass == 0 do return {}

    pass_type := (cursor.pass - 1) % 3
    half_step := pass_step_size(ctx) >> 1
    res := [2]int {
        half_step if pass_type != 0 else 0,
        half_step if pass_type != 1 else 0,
    }
    return res
}

@(private="file")
@(disabled=!ODIN_DEBUG)
validate_context :: proc(using ctx: ^Context)
{
    assert(ctx != nil, "Lightmapper context is null!")
    assert(validation.ctx_initialized, "Context is not initialized!")
    assert(mesh != {}, "Mesh is not set!")
    assert(lightmap != {}, "Target lightmap is not set!")
}

@(private="file")
@(disabled=!ODIN_DEBUG)
validate_shaders :: proc(shaders: Shaders)
{
    assert(shaders.hemi_reduce != nil, "The hemi_reduce compute shader is mandatory!")
    assert(shaders.hemi_weighted_reduce != nil, "The hemi_weighted_reduce compute shader is mandatory!")
    assert(shaders.fullscreen_quad != nil, "The fullscreen_quad vertex shader is mandatory!")
    assert(shaders.blit_lightmap != nil, "The blit_lightmap fragment shader is mandatory!")
}

// Buffer utils
get_vec2f32_from_buffer :: proc(buf: Buffer, idx: i64) -> [2]f32
{
    assert(buf.type != .None)
    addr := rawptr(uintptr(buf.data) + buf.offset + uintptr(idx * auto_cast buf.stride))
    res: [2]f32
    switch buf.type
    {
        case .None: {}
        case .U8:
        {
            res.x = cast(f32)((cast(^[2]u8)addr)[0])
            res.y = cast(f32)((cast(^[2]u8)addr)[1])
        }
        case .U16:
        {
            res.x = cast(f32)((cast(^[2]u16)addr)[0])
            res.y = cast(f32)((cast(^[2]u16)addr)[1])
        }
        case .U32:
        {
            res.x = cast(f32)((cast(^[2]u32)addr)[0])
            res.y = cast(f32)((cast(^[2]u32)addr)[1])
        }
        case .S8:
        {
            res.x = cast(f32)((cast(^[2]i8)addr)[0])
            res.y = cast(f32)((cast(^[2]i8)addr)[1])
        }
        case .S16:
        {
            res.x = cast(f32)((cast(^[2]i16)addr)[0])
            res.y = cast(f32)((cast(^[2]i16)addr)[1])
        }
        case .S32:
        {
            res.x = cast(f32)((cast(^[2]i32)addr)[0])
            res.y = cast(f32)((cast(^[2]i32)addr)[1])
        }
        case .F32:
        {
            res.x = (cast(^[2]f32)addr)[0]
            res.y = (cast(^[2]f32)addr)[1]
        }
    }

    return res
}

get_vec3f32_from_buffer :: proc(buf: Buffer, idx: i64) -> [3]f32
{
    assert(buf.type != .None)
    addr := rawptr(uintptr(buf.data) + buf.offset + uintptr(idx * auto_cast buf.stride))
    res: [3]f32
    switch buf.type
    {
        case .None: {}
        case .U8:
        {
            res.x = cast(f32)((cast(^[3]u8)addr)[0])
            res.y = cast(f32)((cast(^[3]u8)addr)[1])
            res.z = cast(f32)((cast(^[3]u8)addr)[2])
        }
        case .U16:
        {
            res.x = cast(f32)((cast(^[3]u16)addr)[0])
            res.y = cast(f32)((cast(^[3]u16)addr)[1])
            res.z = cast(f32)((cast(^[3]u16)addr)[2])
        }
        case .U32:
        {
            res.x = cast(f32)((cast(^[3]u32)addr)[0])
            res.y = cast(f32)((cast(^[3]u32)addr)[1])
            res.z = cast(f32)((cast(^[3]u32)addr)[2])
        }
        case .S8:
        {
            res.x = cast(f32)((cast(^[3]i8)addr)[0])
            res.y = cast(f32)((cast(^[3]i8)addr)[1])
            res.z = cast(f32)((cast(^[3]i8)addr)[2])
        }
        case .S16:
        {
            res.x = cast(f32)((cast(^[3]i16)addr)[0])
            res.y = cast(f32)((cast(^[3]i16)addr)[1])
            res.z = cast(f32)((cast(^[3]i16)addr)[2])
        }
        case .S32:
        {
            res.x = cast(f32)((cast(^[3]i32)addr)[0])
            res.y = cast(f32)((cast(^[3]i32)addr)[1])
            res.z = cast(f32)((cast(^[3]i32)addr)[2])
        }
        case .F32:
        {
            res.x = (cast(^[3]f32)addr)[0]
            res.y = (cast(^[3]f32)addr)[1]
            res.z = (cast(^[3]f32)addr)[2]
        }
    }

    return res
}

get_i64_from_buffer :: proc(buf: Buffer, idx: i64) -> i64
{
    assert(buf.type != .None)
    addr := rawptr(uintptr(buf.data) + buf.offset + uintptr(idx * auto_cast buf.stride))
    res: i64
    switch buf.type
    {
        case .None: {}
        case .U8:   res = cast(i64)((cast(^u8) addr)^)
        case .U16:  res = cast(i64)((cast(^u16)addr)^)
        case .U32:  res = cast(i64)((cast(^u32)addr)^)
        case .S8:   res = cast(i64)((cast(^i8) addr)^)
        case .S16:  res = cast(i64)((cast(^i16)addr)^)
        case .S32:  res = cast(i64)((cast(^i32)addr)^)
        case .F32:  res = cast(i64)((cast(^f32)addr)^)
    }

    return res
}

// Common utils imported from core libraries
@(private="file")
dot :: la.dot
@(private="file")
cross :: la.cross
@(private="file")
normalize :: la.normalize

@(private="file")
is_inf :: proc(v: [3]f32) -> bool
{
    return math.is_inf(v.x) || math.is_inf(v.y) || math.is_inf(v.z)
}
