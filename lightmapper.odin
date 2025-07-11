
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
import "core:slice"

import sdl "vendor:sdl3"

// General structure and algorithms from: https://github.com/ands/lightmapper

DEBUG_INTERPOLATION :: #config(LM_DEBUG_INTERPOLATION, false)

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
}

destroy_shaders :: proc(device: ^sdl.GPUDevice, shaders: Shaders)
{
    sdl.ReleaseGPUComputePipeline(device, shaders.hemi_reduce)
    sdl.ReleaseGPUComputePipeline(device, shaders.hemi_weighted_reduce)
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
             z_near: f32 = 0.001, z_far: f32 = 100,                 // Hemisphere min/max draw distance.
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

    interp_threshold = interpolation_threshold
    num_passes = 1 + 3 * u32(interpolation_passes)
    hemi_params.size = auto_cast hemisphere_resolution
    hemi_params.z_near = z_near
    hemi_params.z_far  = z_far
    hemi_params.cam_to_surface_distance_modifier = camera_to_surface_distance_modifier
    hemi_params.clear_color = background_color
    hemi_params.batch_count = HEMI_BATCH_TEXTURE_SIZE / (hemi_params.size * [2]u32{ 3, 1 })
    hemi_batch_to_lightmap = make([dynamic][2]u32, hemi_params.batch_count.y * hemi_params.batch_count.x, hemi_params.batch_count.y * hemi_params.batch_count.x)

    // Build GPU Resources

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

    return ctx
}

// Should be called to free resources.
destroy :: proc(using ctx: ^Context)
{
    assert(validation.ctx_initialized, "Attempting to destroy a Lightmapper context without having initialized it first!")
    assert(!validation.iter_begin, "Forgot to call bake_iterate_end! It must be called after each call to bake_iterate_begin (iff it returns true).")
    assert(!validation.bake_begin, "Forgot to call bake_end! It must be called after each call to bake_begin.")

    // CPU resources.
    delete(hemi_batch_to_lightmap)

    // Destroy textures.
    sdl.ReleaseGPUTexture(device, weights_texture)
    sdl.ReleaseGPUTexture(device, hemi_batch_texture)
    sdl.ReleaseGPUTexture(device, hemi_batch_depth_texture)
    sdl.ReleaseGPUTexture(device, hemi_reduce_textures[0])
    sdl.ReleaseGPUTexture(device, hemi_reduce_textures[1])
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
    stride: u32,
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

set_current_mesh :: proc(ctx: ^Context, mesh: Mesh, model_to_world: matrix[4, 4]f32)
{
    ctx.mesh = mesh
    ctx.mesh_transform = model_to_world
    ctx.mesh_normal_mat = la.transpose(la.inverse(model_to_world))

    set_cursor_and_rasterizer(ctx, 0)
}

bake_begin :: proc(using ctx: ^Context, size: [2]u32, format: sdl.GPUTextureFormat)
{
    assert(!validation.bake_begin, "Forgot to call bake_lightmap_end after bake_lightmap_begin!")
    validation.bake_begin = true

    bake_done = false
    lightmap_size = size
    lightmap_format = format

    // Create textures
    final_usages := sdl.GPUTextureUsageFlags { .SAMPLER }
    ensure(sdl.GPUTextureSupportsFormat(device, lightmap_format, .D2, final_usages))
    samples_storage.final_result_texture = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = .R32G32B32A32_FLOAT,
        width = auto_cast size.x,
        height = auto_cast size.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = final_usages,
    })

    samples_usages := sdl.GPUTextureUsageFlags { .COMPUTE_STORAGE_READ }
    ensure(sdl.GPUTextureSupportsFormat(device, lightmap_format, .D2, samples_usages))
    samples_storage.tex = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = lightmap_format,
        width = auto_cast size.x,
        height = auto_cast size.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = samples_usages,
    })

    // Create buffers
    samples_storage.transfer = sdl.CreateGPUTransferBuffer(device, {
        usage = .DOWNLOAD,
        size = size.x * size.y * size_of([4]f32),
    })

    // Allocate CPU resources
    lightmap = make([dynamic][4]f32, size.x * size.y, size.x * size.y)
    samples_storage.uv_map = make([dynamic][2]u32, size.x * size.y, size.x * size.y)

    when DEBUG_INTERPOLATION
    {
        samples_storage.debug_interp_buf = make([dynamic][4]f32, size.x * size.y, size.x * size.y)
    }
}

// NOTE: blit_to is required to have the same size and format as the ones passed to bake_lightmap_begin.
bake_end :: proc(using ctx: ^Context, blit_to: ^sdl.GPUTexture)
{
    assert(validation.bake_begin, "Attempting to call bake_lightmap_end without having called bake_lightmap_begin")
    validation.bake_begin = false

    // Write to the final result texture (which still
    // needs to be blitted into the user texture).
    {
        mapped := sdl.MapGPUTransferBuffer(device, samples_storage.transfer, false)
        intr.mem_copy(mapped, raw_data(lightmap), len(lightmap) * size_of(lightmap[0]))
        sdl.UnmapGPUTransferBuffer(device, samples_storage.transfer)

        pass := sdl.BeginGPUCopyPass(cmd_buf)
        defer sdl.EndGPUCopyPass(pass)

        src := sdl.GPUTextureTransferInfo {
            transfer_buffer = samples_storage.transfer,
            offset = 0,
        }
        dst := sdl.GPUTextureRegion {
            texture = samples_storage.final_result_texture,
            mip_level = 0,
            layer = 0,
            x = 0,
            y = 0,
            z = 0,
            w = lightmap_size.x,
            h = lightmap_size.y,
            d = 1,
        }
        sdl.UploadToGPUTexture(pass, src, dst, false)
    }

    // Blit to user texture.
    {
        blit_info := sdl.GPUBlitInfo {
            source = {
                texture = samples_storage.final_result_texture,
                mip_level = 0,
                layer_or_depth_plane = 0,
                x = 0,
                y = 0,
                w = lightmap_size.x,
                h = lightmap_size.y,
            },
            destination = {
                texture = blit_to,
                mip_level = 0,
                layer_or_depth_plane = 0,
                x = 0,
                y = 0,
                w = lightmap_size.x,
                h = lightmap_size.y,
            },
            load_op = .DONT_CARE,
            clear_color = {},
            flip_mode = .NONE,
            filter = .LINEAR,
            cycle = false,
        }
        sdl.BlitGPUTexture(cmd_buf, blit_info)
    }

    fence_new := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd_buf)
    assert(fence_new != nil)
    ok := sdl.WaitForGPUFences(device, true, &fence, 1)
    assert(ok)
    cmd_buf = sdl.AcquireGPUCommandBuffer(device)

    // Destroy resources.
    sdl.ReleaseGPUTexture(device, samples_storage.final_result_texture)
    samples_storage.final_result_texture = nil
    sdl.ReleaseGPUTexture(device, samples_storage.tex)
    samples_storage.tex = nil
    sdl.ReleaseGPUTransferBuffer(device, samples_storage.transfer)
    samples_storage.transfer = nil
    delete(lightmap)
    lightmap = {}
    delete(samples_storage.uv_map)
    samples_storage.uv_map = {}

    when DEBUG_INTERPOLATION
    {
        delete(samples_storage.debug_interp_buf)
        samples_storage.debug_interp_buf = {}
    }
}

// Optional: Set material characteristics by specifying cos(theta)-dependent weights for incoming light.
// NOTE: This is expensive as this builds and uploads a texture so preferably use it on startup.
Weight_Func_Type :: proc(cos_theta: f32, userdata: rawptr)->f32
set_hemisphere_weights :: proc(using ctx: ^Context,
                               weight_func: Weight_Func_Type,
                               user_data: rawptr,
                               allocator := context.allocator)
{
    sdl.ReleaseGPUTexture(device, weights_texture)

    hemi_size := hemi_params.size
    weights := make([]f32, hemi_size * hemi_size * 2 * 3, allocator = allocator)
    defer delete(weights)

    center  := f32(hemi_size - 1) * 0.5
    sum     := 0.0
    for y in 0..<hemi_size
    {
        // In SDL_GPU +y is down (doesn't matter because we only
        // need its absolute value, but for future reference...)
        dy := -(f32(y) - center) / (f32(hemi_size) * 0.5)
        for x in 0..<hemi_size
        {
            dx := (f32(x) - center) / (f32(hemi_size) * 0.5)
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
    intr.mem_copy(transfer_dst, raw_data(weights), len(weights) * size_of(f32))
    sdl.UnmapGPUTransferBuffer(device, transfer_buf)

    pass := sdl.BeginGPUCopyPass(cmd_buf)
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
}

// This describes how your scene should be rendered
// for correct/optimal (depending on the parameter) behavior.
Scene_Render_Params :: struct
{
    depth_only: bool,        // Optional, this is to speed the first pass up.
    render_shadowmap: bool,  // Optional, if you want to include a directional light source.

    viewport_offset: [2]u32,
    viewport_size:   [2]u32,
    // NOTE: Assuming left-handed coordinate system.
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
    assert(validation.bake_begin, "Forgot to call bake_begin!")

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

        triangles_left := cursor.tri_base_idx + 3 < mesh.tri_count * 3
        if triangles_left
        {
            // Move onto the next triangle.
            set_cursor_and_rasterizer(ctx, auto_cast cursor.tri_base_idx + 3)
        }
        else
        {
            // End of pass.
            when DEBUG_INTERPOLATION
            {
                fmt.println("pass", cursor.pass, "done! Stats:")
                fmt.println("Num interpolated:", samples_storage.num_interpolated, "Num_sampled:", samples_storage.num_sampled)
                samples_storage.num_interpolated = 0
                samples_storage.num_sampled = 0
            }

            cursor.pass += 1

            if bake_pass != nil
            {
                sdl.EndGPURenderPass(bake_pass)
                bake_pass = nil
            }

            integrate_hemi_batch_and_copy_to_transfer_buf(ctx)
            cursor.hemi_idx = 0
            read_back_samples_texture(ctx)

            if cursor.pass < num_passes
            {
                if cursor.hemi_idx > 0 && cursor.hemi_side > 0 {
                    begin_bake_render_pass(ctx, true)
                }

                // Start the next pass.
                set_cursor_and_rasterizer(ctx, 0)
            }
            else
            {
                // We've finished the lightmapping process.
                bake_done = true

                // Reset baking state.
                cursor.hemi_idx = 0
                cursor.pass = 0
                cursor.hemi_side = 5

                validation.iter_begin = false
                return {}, false
            }
        }
    }

    // Prepare hemisphere.
    if cursor.hemi_side == 0
    {
        // Prepare hemisphere batch.
        if cursor.hemi_idx == 0
        {
            assert(bake_pass == nil)  // Pass should have ended by now.

            submit_and_recreate_cmd_buf(ctx)

            begin_bake_render_pass(ctx)
        }

        hemi_batch_to_lightmap[cursor.hemi_idx] = rasterizer.pos
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
    if !validation.iter_begin && bake_done do return 1.0

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
    assert(validation.bake_begin, "Forgot to call bake_begin!")
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

            integrate_hemi_batch_and_copy_to_transfer_buf(ctx)
            cursor.hemi_idx = 0
        }
    }
}

// Post-processing functions.

// This postprocessing step is important because it eliminates
// "invalid" pixels, obtained by rendering the back-side of objects.
postprocess_dilate :: proc(using ctx: ^Context)
{
    assert(validation.bake_begin, "Forgot to call bake_begin!")
    validate_context(ctx)

    output := slice.clone_to_dynamic(lightmap[:])
    defer
    {
        delete(lightmap)
        lightmap = output
    }

    for y in 0..<lightmap_size.y
    {
        for x in 0..<lightmap_size.x
        {
            lm_idx := y * lightmap_size.y + x
            output_color := lightmap[lm_idx]
            if output_color == { 0, 0, 0, 0 }
            {
                n := 0
                offsets := [][2]int { {-1, 0}, {0, 1}, {1, 0}, {0, -1} }
                for offset_idx in 0..<4
                {
                    sample_idx := offsets[offset_idx] + {auto_cast x, auto_cast y}
                    in_bounds_x := sample_idx.x >= 0 && auto_cast sample_idx.x < lightmap_size.x
                    in_bounds_y := sample_idx.y >= 0 && auto_cast sample_idx.y < lightmap_size.y
                    if in_bounds_x && in_bounds_y
                    {
                        sample := lightmap[sample_idx.y * auto_cast lightmap_size.x + sample_idx.x]
                        if sample != { 0, 0, 0, 0 }
                        {
                            output_color += sample
                            n += 1
                        }
                    }
                }

                if n > 0
                {
                    output_color *= 1.0 / f32(n)
                }
            }

            output[lm_idx] = output_color
        }
    }
}

postprocess_box_blur :: proc(using ctx: ^Context)
{
    assert(validation.bake_begin, "Forgot to call bake_begin!")
    validate_context(ctx)

    output := slice.clone_to_dynamic(lightmap[:])
    defer
    {
        delete(lightmap)
        lightmap = output
    }

    for y in 0..<lightmap_size.y
    {
        for x in 0..<lightmap_size.x
        {
            lm_idx := y * lightmap_size.y + x
            output_color: [4]f32

            n := 0

            for offset_y in -1..=1
            {
                for offset_x in -1..=1
                {
                    sample_idx := [2]int {offset_x, offset_y} + {int(x), int(y)}
                    in_bounds_x := sample_idx.x >= 0 && sample_idx.x < int(lightmap_size.x)
                    in_bounds_y := sample_idx.y >= 0 && sample_idx.y < int(lightmap_size.y)
                    if in_bounds_x && in_bounds_y
                    {
                        sample := lightmap[sample_idx.y * int(lightmap_size.x) + sample_idx.x]
                        if sample != { 0, 0, 0, 0 }
                        {
                            output_color += sample
                            n += 1
                        }
                    }
                }
            }

            output[y * lightmap_size.x + x] = output_color / f32(n) if n > 0 else 0.0
        }
    }
}

////////////////////////////
// Internal

// A good introduction to learn some of the theory behind this:
// http://the-witness.net/news/2010/09/hemicube-rendering-and-integration/

Context :: struct
{
    // Settings
    num_passes: u32,
    interp_threshold: f32,
    do_log_stats: bool,

    // Validation
    validation: Validation,

    // Bound state
    mesh: Mesh,
    mesh_transform:  matrix[4, 4]f32,
    mesh_normal_mat: matrix[4, 4]f32,
    lightmap_size: [2]u32,
    lightmap_format: sdl.GPUTextureFormat,

    // Lightmap baking state
    cursor: Cursor,
    rasterizer: Rasterizer,
    tri_sample: Tri_Sample,
    hemi_params: Hemisphere_Params,
    hemi_batch_to_lightmap: [dynamic][2]u32,  // hemisphere idx in batch -> lightmap pixel
    samples_storage: Samples_Storage,
    // We need to keep a copy on the CPU for irradiance caching.
    // buf is a buffer which contains packed hemisphere
    // sample results. This data is then written into the lightmap
    // using a coordinate transformation (map).
    lightmap: [dynamic][4]f32,
    bake_done: bool,

    // GPU Resources
    device: ^sdl.GPUDevice,
    cmd_buf: ^sdl.GPUCommandBuffer,
    fence: ^sdl.GPUFence,  // For double buffering.
    bake_pass: ^sdl.GPURenderPass,
    shaders: Shaders,

    // Textures
    weights_texture: ^sdl.GPUTexture,  // Holds the weights to convert pixels from hemicube to hemisphere.
    hemi_batch_texture: ^sdl.GPUTexture,  // Holds many hemispheres (which in turn hold 5 different hemicube sides each)
    hemi_batch_depth_texture: ^sdl.GPUTexture,
    hemi_reduce_textures: [2]^sdl.GPUTexture,  // Ping-pong buffers used to run hemisphere reduction steps.
}

Validation :: struct
{
    ctx_initialized: bool,
    bake_begin: bool,
    iter_begin: bool,
}

Cursor :: struct
{
    pass: u32,
    tri_base_idx: int,  // The index of the first vertex.
    tri_verts_pos: [3] [3]f32,
    tri_verts_normal: [3] [3]f32,
    tri_verts_lm_uv: [3] [2]f32,

    hemi_idx:  u32,  // Index in the current batch.
    hemi_side: u32,  // [0, 4], side of the hemicube.
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
    min: [2]u32,
    max: [2]u32,
    pos: [2]u32,
}

Samples_Storage :: struct
{
    // Final texture which gets then copied into the user texture.
    // This is required because the user texture in general is sampleable
    // (to be able to get multiple bounces) which sadly makes it incompatible
    // with the compute pipeline in general.
    final_result_texture: ^sdl.GPUTexture,

    transfer: ^sdl.GPUTransferBuffer,
    uv_map: [dynamic][2]u32,
    tex: ^sdl.GPUTexture,  // Final lightmap
    uv_write_pos: u32,

    using debug: Debug_Samples_Storage,
}

when DEBUG_INTERPOLATION
{
    Debug_Samples_Storage :: struct
    {
        debug_interp_buf: [dynamic][4]f32,
        num_interpolated: u32,
        num_sampled: u32,
    }
}
else
{
    Debug_Samples_Storage :: struct {}
}

HEMI_BATCH_TEXTURE_SIZE :: [2]u32 { 512 * 3, 512 }
#assert(HEMI_BATCH_TEXTURE_SIZE.x % 3 == 0 && intr.count_ones(HEMI_BATCH_TEXTURE_SIZE.x / 3) == 1,
        "HEMI_BATCH_TEXTURE_SIZE.x must be 3 * power-of-two!")
#assert(intr.count_ones(HEMI_BATCH_TEXTURE_SIZE.y) == 1,
        "HEMI_BATCH_TEXTURE_SIZE.y must be a power of two!")

Hemisphere_Params :: struct
{
    z_near: f32,
    z_far: f32,
    size: u32,
    cam_to_surface_distance_modifier: f32,
    clear_color: [3]f32, // Do i actually need this?
    batch_count: [2]u32,
}

compute_current_camera :: proc(using ctx: ^Context) -> (viewport_offset: [2]u32, viewport_size: [2]u32, world_to_view: matrix[4, 4]f32, view_to_proj: matrix[4, 4]f32)
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
            world_to_view, view_to_proj = compute_matrices(pos, -up, -dir, -zn, zn, 0, zn, zn, zf)
            viewport_offset = { x + size + size, y }
            viewport_size   = { size, size / 2.0 }
        }
        case 4:  // Up
        {
            world_to_view, view_to_proj = compute_matrices(pos, up, dir, -zn, zn, -zn, 0, zn, zf)
            viewport_offset = { x + size + size, y + size / 2 }
            viewport_size   = { size, size / 2.0 }
        }
    }

    return viewport_offset, viewport_size, world_to_view, view_to_proj

    compute_matrices :: proc(pos: [3]f32, dir: [3]f32, up: [3]f32,
                             l: f32, r: f32, b: f32, t: f32, n: f32, f: f32) ->
                             (world_to_view: matrix[4, 4]f32, view_to_proj: matrix[4, 4]f32)
    {
        assert(abs(dot(dir, up)) < 0.001)  // Should be perpendicular
        assert(abs(la.length(dir) - 1.0) < 0.001)  // Should be normalized
        assert(abs(la.length(up)  - 1.0) < 0.001)  // Should be normalized

        right := cross(up, dir)
        world_to_view = {
            right.x, right.y, right.z, dot(right, -pos),
            up.x,    up.y,    up.z,    dot(up, -pos),
            dir.x,   dir.y,   dir.z,   dot(dir, -pos),
            0,       0,       0,       1.0,
        }

        // Perspective view from frustum.
        view_to_proj = {
            2.0 * n / (r - l), 0,                 (r + l) / (r - l),  0,
            0,                 2.0 * n / (t - b), (t + b) / (t - b),  0,
            0,                 0,                 (f + n) / (f - n),  -2.0 * f * n / (f - n),
            0,                 0,                 1,                  0,
        }

        return world_to_view, view_to_proj
    }
}

set_cursor_and_rasterizer :: proc(using ctx: ^Context, tri_idx: i64)
{
    cursor.tri_base_idx = auto_cast tri_idx

    verts_indices: [3]i64 = tri_idx
    if mesh.use_indices {
        for i in 0..<3 do verts_indices[i] = get_i64_from_buffer(mesh.indices, tri_idx + auto_cast i)
    } else {
        for i in 0..<3 do verts_indices[i] = tri_idx + auto_cast i
    }

    uv_scale := [2]f32 { auto_cast lightmap_size.x, auto_cast lightmap_size.y }

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
        cursor.tri_verts_normal[i] = (mesh_normal_mat * normal).xyz

        cursor.tri_verts_lm_uv[i] *= uv_scale

        uv_min = { min(uv_min.x, cursor.tri_verts_lm_uv[i].x), min(uv_min.y, cursor.tri_verts_lm_uv[i].y) }
        uv_max = { max(uv_max.x, cursor.tri_verts_lm_uv[i].x), max(uv_max.y, cursor.tri_verts_lm_uv[i].y) }
    }

    // Calculate bounding box on lightmap for conservative rasterization.
    bb_min := la.floor(uv_min)
    bb_max := la.ceil(uv_max)
    rasterizer.min.x = max(u32(bb_min.x), 0)
    rasterizer.min.y = max(u32(bb_min.y), 0)
    rasterizer.max.x = min(u32(bb_max.x) + 1, u32(lightmap_size.x - 1))
    rasterizer.max.y = min(u32(bb_max.y) + 1, u32(lightmap_size.y - 1))
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
integrate_hemi_batch_and_copy_to_transfer_buf :: proc(using ctx: ^Context)
{
    if cursor.hemi_idx == 0 do return

    tex_target_idx := 0
    tex_read_idx   := 1

    reduced_size := hemi_params.size

    // NOTE: Coupled to the shader code.
    Uniforms :: struct
    {
        input_size: [2]u32,
        output_size: [2]u32,
    }

    // First pass: Downsample and apply weights.
    {
        THREAD_GROUP_SIZE: u32 : 8  // NOTE: Coupled to the shader code.
        samples_per_group := 1 * THREAD_GROUP_SIZE
        assert(hemi_params.size > max(hemi_params.batch_count.x, hemi_params.batch_count.y))

        group_count: [2]u32 = hemi_params.size / samples_per_group * hemi_params.batch_count
        group_count = { max(group_count.x, hemi_params.batch_count.x), max(group_count.y, hemi_params.batch_count.y) }

        uniforms := Uniforms {
            input_size = reduced_size * hemi_params.batch_count,
            output_size = group_count,
        }
        sdl.PushGPUComputeUniformData(cmd_buf, 0, &uniforms, size_of(uniforms))

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

        assert(group_count.x > 0 && group_count.y > 0)
        sdl.DispatchGPUCompute(pass, group_count.x, group_count.y, 1)

        reduced_size /= samples_per_group
    }

    // Swap back textures to undo last iteration.
    tex_target_idx = tex_read_idx
    tex_read_idx = (tex_target_idx + 1) % 2

    // Successive passes: non-weighted downsampling passes.
    for reduced_size > 1
    {
        THREAD_GROUP_SIZE: u32 : 8  // NOTE: Coupled to the shader code.
        num_samples_per_group := 2 * THREAD_GROUP_SIZE

        group_count: [2]u32 = reduced_size / num_samples_per_group * hemi_params.batch_count
        group_count = { max(group_count.x, hemi_params.batch_count.x), max(group_count.y, hemi_params.batch_count.y) }

        uniforms := Uniforms {
            input_size = reduced_size * hemi_params.batch_count,
            output_size = group_count,
        }
        sdl.PushGPUComputeUniformData(cmd_buf, 0, &uniforms, size_of(uniforms))

        // Read from hemi_reduce_textures[read] and write to hemi_reduce_textures[target].
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

        assert(group_count.x > 0 && group_count.y > 0)
        sdl.DispatchGPUCompute(pass, group_count.x, group_count.y, 1)

        // Swap textures.
        tex_target_idx = tex_read_idx
        tex_read_idx = (tex_target_idx + 1) % 2

        reduced_size /= num_samples_per_group
    }

    // Swap back textures to undo last iteration.
    tex_target_idx = tex_read_idx
    tex_read_idx = (tex_target_idx + 1) % 2

    // Copy results to transfer buffer.
    {
        pass := sdl.BeginGPUCopyPass(cmd_buf)
        defer sdl.EndGPUCopyPass(pass)

        // TODO: Results in the optimized version of this
        // are slightly different? Investigate this...
        when false
        {
            // Slow version, pixel by pixel copy.

            for dst_pixel, idx in hemi_batch_to_lightmap
            {
                if auto_cast idx >= cursor.hemi_idx do break

                src_x := u32(idx) % hemi_params.batch_count.x
                src_y := u32(idx) / hemi_params.batch_count.y

                src := sdl.GPUTextureRegion {
                    texture = hemi_reduce_textures[tex_target_idx],
                    mip_level = 0,
                    layer = 0,
                    x = src_x,
                    y = src_y,
                    z = 0,
                    w = 1,
                    h = 1,
                    d = 1,
                }
                dst := sdl.GPUTextureTransferInfo {
                    transfer_buffer = samples_storage.transfer,
                    offset = (samples_storage.uv_write_pos + auto_cast idx) * size_of([4]f32),
                }
                sdl.DownloadFromGPUTexture(pass, src, dst)
            }

            // Update uv map.
            for i in 0..<cursor.hemi_idx {
                samples_storage.uv_map[samples_storage.uv_write_pos + i] = hemi_batch_to_lightmap[i]
            }

            samples_storage.uv_write_pos += cursor.hemi_idx
        }
        else
        {
            // Fast version, batched copies.

            write_pos_old := samples_storage.uv_write_pos

            if cursor.hemi_idx / hemi_params.batch_count.x > 0
            {
                total_hemis := hemi_params.batch_count.x * hemi_params.batch_count.y
                copy_size := hemi_params.batch_count
                if cursor.hemi_idx < total_hemis {
                    copy_size.y = cursor.hemi_idx / hemi_params.batch_count.x
                }

                src := sdl.GPUTextureRegion {
                    texture = hemi_reduce_textures[tex_target_idx],
                    mip_level = 0,
                    layer = 0,
                    x = 0,
                    y = 0,
                    z = 0,
                    w = copy_size.x,
                    h = copy_size.y,
                    d = 1,
                }
                dst := sdl.GPUTextureTransferInfo {
                    transfer_buffer = samples_storage.transfer,
                    offset = samples_storage.uv_write_pos * size_of([4]f32),
                }
                sdl.DownloadFromGPUTexture(pass, src, dst)

                written := copy_size.y * copy_size.x
                for i, idx in samples_storage.uv_write_pos..<samples_storage.uv_write_pos+written {
                    samples_storage.uv_map[i] = hemi_batch_to_lightmap[idx]
                }

                samples_storage.uv_write_pos += written
            }

            // Copy the last partial row if it exists (most of the time it doesn't).
            if cursor.hemi_idx % hemi_params.batch_count.x > 0
            {
                remaining_copy_size := [2]u32 { cursor.hemi_idx % hemi_params.batch_count.x, 1 }
                src := sdl.GPUTextureRegion {
                    texture = hemi_reduce_textures[tex_target_idx],
                    mip_level = 0,
                    layer = 0,
                    x = 0,
                    y = cursor.hemi_idx / hemi_params.batch_count.x,
                    z = 0,
                    w = remaining_copy_size.x,
                    h = remaining_copy_size.y,
                    d = 1,
                }
                dst := sdl.GPUTextureTransferInfo {
                    transfer_buffer = samples_storage.transfer,
                    offset = samples_storage.uv_write_pos * size_of([4]f32),
                }
                sdl.DownloadFromGPUTexture(pass, src, dst)

                written := remaining_copy_size.y * remaining_copy_size.x
                for i, idx in samples_storage.uv_write_pos..<samples_storage.uv_write_pos+written {
                    samples_storage.uv_map[i] = hemi_batch_to_lightmap[idx]
                }

                samples_storage.uv_write_pos += written
            }

            assert(write_pos_old + cursor.hemi_idx == samples_storage.uv_write_pos)
        }
    }
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
    for rasterizer.pos.x >= rasterizer.max.x && !has_rasterizer_finished(ctx)
    {
        rasterizer.pos.x = rasterizer.min.x + pass_offset(ctx).x
        rasterizer.pos.y += step
    }
}

// If it returns true, ctx.tri_sample will be filled in.
try_sampling_rasterizer_position :: proc(using ctx: ^Context) -> bool
{
    if has_rasterizer_finished(ctx) do return false

    lm_already_set := lightmap[rasterizer.pos.y * lightmap_size.x + rasterizer.pos.x] != { 0, 0, 0, 0 }
    if lm_already_set do return false

    // Try computing centroid by clipping the pixel against the triangle.
    raster_pos := [2]f32 { auto_cast rasterizer.pos.x, auto_cast rasterizer.pos.y }
    clipped_array, clipped_len := pixel_tri_clip(raster_pos, cursor.tri_verts_lm_uv)
    if clipped_len <= 0 do return false  // Nothing left.

    clipped := clipped_array[:clipped_len]

    // Compute centroid position and area.
    // http://the-witness.net/news/2010/09/hemicube-rendering-and-integration/
    // Centroid sampling basically makes it so we don't clip inside of a wall
    // for hemisphere rendering of an intersecting floor. (This only works if
    // geometry is well-formed and there are no intersections)
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
    area = abs(area / 2.0)

    if area <= 0.0 do return false  // No area left.

    // Compute barycentric coords.
    uv := to_barycentric(cursor.tri_verts_lm_uv[0], cursor.tri_verts_lm_uv[1], cursor.tri_verts_lm_uv[2], centroid)
    if math.is_nan(uv.x) || math.is_inf(uv.x) || math.is_nan(uv.y) || math.is_inf(uv.y) {
        return false // Degenerate case.
    }

    // Try to interpolate color from neighbors.
    // (Irradiance caching)
    if cursor.pass > 0
    {
        neighbors: [4][4]f32
        neighbor_count := 0
        neighbors_expected := 0

        d := pass_step_size(ctx) / 2
        dirs := ((cursor.pass - 1) % 3) + 1
        if dirs & 1 != 0  // Check x-neighbors with distance d
        {
            neighbors_expected += 2
            if rasterizer.pos.x - d >= rasterizer.min.x &&
               rasterizer.pos.x + d <= rasterizer.max.x
            {
                neighbors[neighbor_count + 0] = lightmap[rasterizer.pos.y * auto_cast lightmap_size.x + (rasterizer.pos.x - d)]
                neighbors[neighbor_count + 1] = lightmap[rasterizer.pos.y * auto_cast lightmap_size.x + (rasterizer.pos.x + d)]
                neighbor_count += 2
            }
        }
        if dirs & 2 != 0  // Check y-neighbors with distance d
        {
            neighbors_expected += 2
            if rasterizer.pos.y - d >= rasterizer.min.y &&
               rasterizer.pos.y + d <= rasterizer.max.y
            {
                neighbors[neighbor_count + 0] = lightmap[(rasterizer.pos.y - d) * auto_cast lightmap_size.x + rasterizer.pos.x]
                neighbors[neighbor_count + 1] = lightmap[(rasterizer.pos.y + d) * auto_cast lightmap_size.x + rasterizer.pos.x]
                neighbor_count += 2
            }
        }

        if neighbor_count == neighbors_expected
        {
            avg: [4]f32
            for neighbor_idx in 0..<neighbor_count
            {
                avg += neighbors[neighbor_idx]
            }

            avg /= f32(neighbor_count)

            interpolate := true
            for neighbor_idx in 0..<neighbor_count
            {
                is_zero := true
                for channel in 0..<4
                {
                    neighbor := neighbors[neighbor_idx]
                    if neighbor[channel] != 0.0 {
                        is_zero = false
                    }
                    if abs(neighbor[channel] - avg[channel]) > interp_threshold {
                        interpolate = false
                    }
                }

                if is_zero do interpolate = false
                if !interpolate do break
            }

            if interpolate
            {
                lightmap[rasterizer.pos.y * auto_cast lightmap_size.x + rasterizer.pos.x] = avg

                when DEBUG_INTERPOLATION {  // Write green pixel.
                    samples_storage.debug_interp_buf[rasterizer.pos.y * auto_cast lightmap_size.x + rasterizer.pos.x] = { 0, 1, 0, 1 }
                    samples_storage.num_interpolated += 1
                }

                return false
            }
        }
    }

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
    // Pseudo-random directions fix banding artifacts, though they increase noise.
    // Noise is much easier to deal with, by smoothing the lightmap a bit.
    when false
    {
        tri_sample.up = normalize(cross(up, tri_sample.dir))
    }
    else
    {
        tri_side := normalize(cross(up, tri_sample.dir))
        tri_up   := normalize(cross(tri_sample.dir, tri_side))
        rx := rasterizer.pos.x % 3
        ry := rasterizer.pos.y % 3

        // TODO: The 3x3 pattern does not seem to work well here...
        // investigate why?

        base_angle: f32 = 0.1
        base_angles := [3][3]f32 {
            { base_angle, base_angle + 1.0 / 3.0, base_angle + 2.0 / 3.0 },
            { base_angle + 1.0 / 3.0, base_angle + 2.0 / 3.0, base_angle },
            { base_angle + 2.0 / 3.0, base_angle, base_angle + 1.0 / 3.0 }
        }
        phi := 2.0 * math.PI * base_angles[ry][rx] + 0.1 * rand.float32()

        tri_sample.up = normalize(tri_side * math.cos(phi) + tri_up * math.sin(phi))
    }

    when DEBUG_INTERPOLATION {  // Write red pixel.
        samples_storage.debug_interp_buf[rasterizer.pos.y * auto_cast lightmap_size.x + rasterizer.pos.x] = { 1, 0, 0, 1 }
        samples_storage.num_sampled += 1
    }

    return true
}

pixel_tri_clip :: proc(pixel_top_left: [2]f32, tri: [3][2]f32) -> (res: [16][2]f32, num: int)
{
    pixel_poly := [16][2]f32 {
        pixel_top_left + { 0, 0 },
        pixel_top_left + { 1, 0 },
        pixel_top_left + { 1, 1 },
        pixel_top_left + { 0, 1 },
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    }

    n_poly := 4
    num = n_poly
    dir := left_of(tri[0], tri[1], tri[2])
    for i in 0..<3
    {
        if num == 0 do break

        j := i - 1 if i > 0 else 2

        if i != 0
        {
            for n_poly = 0; n_poly < num; n_poly += 1 {
                pixel_poly[n_poly] = res[n_poly]
            }
        }

        num = 0
        v0 := pixel_poly[n_poly - 1]
        side_0 := left_of(tri[j], tri[i], v0)
        if side_0 != -dir
        {
            res[num] = v0
            num += 1
        }

        for k in 0..<n_poly
        {
            v1 := pixel_poly[k]
            side_1 := left_of(tri[j], tri[i], v1)
            intersect, inter_p := line_intersection(tri[j], tri[i], v0, v1)
            if side_0 + side_1 == 0 && side_0 != 0 && intersect {
                res[num] = inter_p
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
        res := cross(b - a, c - b)
        if res < 0 do return -1
        if res > 0 do return +1
        return 0
    }

    line_intersection :: proc(x0: [2]f32, x1: [2]f32, y0: [2]f32, y1: [2]f32) -> (intersect: bool, p: [2]f32)
    {
        dx := x1 - x0
        dy := y1 - y0
        d := x0 - y0
        dyx := cross(dy, dx)
        if dyx == 0.0 do return false, {}

        dyx = cross(d, dx) / dyx
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

pass_step_size :: proc(using ctx: ^Context) -> u32
{
    pass_minus_one := cursor.pass - 1 if cursor.pass > 0 else 0
    shift := u32(num_passes / 3 - pass_minus_one / 3)
    step: u32 = 1 << shift
    assert(step > 0)
    return step
}

pass_offset :: proc(using ctx: ^Context) -> [2]u32
{
    if cursor.pass <= 0 do return { 0, 0 }

    pass_type := (cursor.pass - 1) % 3
    half_step := pass_step_size(ctx) >> 1
    return {
        half_step if pass_type != 1 else 0,
        half_step if pass_type != 0 else 0,
    }
}

read_back_samples_texture :: proc(using ctx: ^Context)
{
    fence_new := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd_buf)

    assert(fence_new != nil)
    wait_on := []^sdl.GPUFence { fence, fence_new }
    ok := sdl.WaitForGPUFences(device, true, raw_data(wait_on), auto_cast len(wait_on))
    assert(ok)
    cmd_buf = sdl.AcquireGPUCommandBuffer(device)

    mapped := sdl.MapGPUTransferBuffer(device, samples_storage.transfer, false)
    mapped_typed := cast([^][4]f32)mapped
    for pos in 0..<samples_storage.uv_write_pos
    {
        lm_uv := samples_storage.uv_map[pos]
        pixel := mapped_typed[pos]
        validity := pixel[3]
        cur_lm_pixel := lightmap[lm_uv.y * lightmap_size.x + lm_uv.x]
        if cur_lm_pixel == { 0, 0, 0, 0 } && validity > 0.9
        {
            scale := 1.0 / validity
            addr := &lightmap[lm_uv.y * lightmap_size.x + lm_uv.x]
            addr^ = pixel * scale
            addr^[0] = max(addr^[0], 1.175494e-38)
            addr^[1] = max(addr^[1], 1.175494e-38)
            addr^[2] = max(addr^[2], 1.175494e-38)
            addr^[3] = 1.0
        }
    }

    sdl.UnmapGPUTransferBuffer(device, samples_storage.transfer)

    samples_storage.uv_write_pos = 0
}

@(private="file")
submit_and_recreate_cmd_buf :: proc(using ctx: ^Context)
{
    fence_new := sdl.SubmitGPUCommandBufferAndAcquireFence(cmd_buf)
    assert(fence_new != nil)
    if fence != nil
    {
        ok := sdl.WaitForGPUFences(device, true, &fence, 1)
        assert(ok)
        sdl.ReleaseGPUFence(device, fence)

    }
    fence = fence_new
    cmd_buf = sdl.AcquireGPUCommandBuffer(device)
}

@(private="file")
begin_bake_render_pass :: proc(using ctx: ^Context, keep_old_content := false)
{
    color_target := sdl.GPUColorTargetInfo {
        texture = hemi_batch_texture,
        clear_color = { hemi_params.clear_color.x, hemi_params.clear_color.y, hemi_params.clear_color.z, 1.0 },
        load_op = .LOAD if keep_old_content else .CLEAR,
        store_op = .STORE,
    }
    depth_target := sdl.GPUDepthStencilTargetInfo {
        texture = hemi_batch_depth_texture,
        clear_depth = 1.0,
        load_op = .LOAD if keep_old_content else .CLEAR,
        store_op = .STORE,
        stencil_load_op = .DONT_CARE,
        stencil_store_op = .DONT_CARE,
    }
    bake_pass = sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, &depth_target)
}

@(private="file")
@(disabled=!ODIN_DEBUG)
validate_context :: proc(using ctx: ^Context)
{
    assert(ctx != nil, "Lightmapper context is null!")
    assert(validation.ctx_initialized, "Context is not initialized!")
    assert(mesh != {}, "Mesh is not set!")
    validate_shaders(shaders)
}

@(private="file")
@(disabled=!ODIN_DEBUG)
validate_shaders :: proc(shaders: Shaders)
{
    assert(shaders.hemi_reduce != nil, "The hemi_reduce compute shader is mandatory!")
    assert(shaders.hemi_weighted_reduce != nil, "The hemi_weighted_reduce compute shader is mandatory!")
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
