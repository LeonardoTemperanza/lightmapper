
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

package main

import "core:fmt"
import intr "base:intrinsics"
import "core:math"
import "core:math/linalg"
import "core:mem"
import "core:os"
import "core:c"

import stbi "vendor:stb/image"
import sdl "vendor:sdl3"
import lm "../"

BACKGROUND_COLOR :: [3]f32 { 0.6, 0.5, 0.7 }

LIGHTMAP_SIZE :: [2]u32 { 654, 654 }
LIGHTMAP_FORMAT :: sdl.GPUTextureFormat.R16G16B16A16_FLOAT

main :: proc()
{
    window, device := init_sdl()
    ensure(window != nil && device != nil)
    defer quit_sdl(window, device)

    ts_freq := sdl.GetPerformanceFrequency()

    mesh := upload_mesh(device, MESH_VERTS, MESH_INDICES)
    defer cleanup_mesh(device, &mesh)

    skysphere, sky_tex := upload_sky_resources(device)
    defer cleanup_sky_resources(device, &skysphere, &sky_tex)

    // Pipelines must be compatible with the lightmapping passes (hemisphere target formats that you provide)
    pipelines := make_pipelines(device, window)
    defer cleanup_pipelines(device, &pipelines)

    // There are many different strategies for loading shaders
    // in SDL_GPU, so this library leaves this responsibility
    // to the user. The shaders can be found in this repository.
    // This example will just use precompiled shaders embedded
    // into the executable (in SPIRV and MSL formats, for windows,
    // linux and mac).
    lm_shaders := make_lm_shaders(device)
    defer lm.destroy_shaders(device, lm_shaders)

    _lm_ctx := lm.init(device, lm_shaders, LIGHTMAP_FORMAT, get_depth_format(device),
                       background_color = BACKGROUND_COLOR,
                       interpolation_passes = 2,
                       hemisphere_resolution = 256)
    lm_ctx  := &_lm_ctx
    defer lm.destroy(lm_ctx)

    linear_sampler := sdl.CreateGPUSampler(device, {
        min_filter = .LINEAR,
        mag_filter = .LINEAR,
        mipmap_mode = .LINEAR,
        address_mode_u = .REPEAT,
        address_mode_v = .REPEAT,
        address_mode_w = .REPEAT,
        max_lod = 1000,
    })
    defer sdl.ReleaseGPUSampler(device, linear_sampler)

    lightmap := lm.make_lightmap(device, LIGHTMAP_SIZE, LIGHTMAP_FORMAT)
    defer lm.destroy_lightmap(device, lightmap)

    // Specify your buffer formats.
    mesh_info := lm.Mesh {
        positions = {
            data = raw_data(MESH_VERTS),
            type = .F32,
            stride = size_of(Vertex),
            offset = offset_of(Vertex, pos)
        },
        normals = {
            data = raw_data(MESH_VERTS),
            type = .F32,
            stride = size_of(Vertex),
            offset = offset_of(Vertex, normal),
        },
        lm_uvs = {
            data = raw_data(MESH_VERTS),
            type = .F32,
            stride = size_of(Vertex),
            offset = offset_of(Vertex, lm_uv),
        },
        indices = {
            data = raw_data(MESH_INDICES),
            type = .U32,
            stride = size_of(u32),
            offset = 0,
        },
        tri_count = len(MESH_INDICES) / 3,
        use_indices = true,
    }

    // Build lightmap.
    {
        fmt.println("Started to bake lightmap...")
        bake_begin_ts := sdl.GetPerformanceCounter()

        NUM_BOUNCES :: 5  // 1 for ambient occlusion only, 2 or more for global illumination.
        for bounce in 0..<NUM_BOUNCES
        {
            lm.bake_begin(lm_ctx, LIGHTMAP_SIZE, LIGHTMAP_FORMAT)

            model_to_world: matrix[4, 4]f32 = 1
            lm.set_current_mesh(lm_ctx, mesh_info, model_to_world)

            for render_params in lm.bake_iterate_begin(lm_ctx)
            {
                defer lm.bake_iterate_end(lm_ctx)

                cmd_buf := render_params.cmd_buf

                render_scene(cmd_buf, render_params, lightmap, linear_sampler, pipelines, mesh, skysphere, sky_tex)

                @(static) print_counter := 0
                if print_counter > 20000
                {
                    fmt.printf("Bounce %v, progress: %.6f%% \r", bounce, lm.bake_progress(lm_ctx))
                    print_counter = 0
                }
                print_counter += 1
            }

            fmt.printf("Bounce %v, progress: %.6f%% \r", bounce, lm.bake_progress(lm_ctx))
            fmt.println("")

            // Post-process the lightmap as you wish.
            for i in 0..<16
            {
                lm.postprocess_dilate(lm_ctx)
                lm.postprocess_dilate(lm_ctx)
            }
            lm.postprocess_box_blur(lm_ctx)
            lm.postprocess_dilate(lm_ctx)

            lm.bake_end(lm_ctx, lightmap)
        }

        bake_end_ts := sdl.GetPerformanceCounter()
        elapsed := f32(f64((bake_end_ts - bake_begin_ts)*1000) / f64(ts_freq)) / 1000.0

        when ODIN_DEBUG {
            fmt.printfln("Done with lightmap baking! (%.6fs) (DEBUG BUILD)", elapsed)
        } else {
            fmt.printfln("Done with lightmap baking! (%.6fs)", elapsed)
        }
    }

    sdl.ShowWindow(window)
    view_results(window, device, lightmap, pipelines, mesh, skysphere, sky_tex)
}

view_results :: proc(window: ^sdl.Window, device: ^sdl.GPUDevice, lm_tex: ^sdl.GPUTexture, pipelines: Pipelines, mesh: Mesh_GPU, skysphere: Mesh_GPU, sky_tex: ^sdl.GPUTexture)
{
    fmt.println("A view of the result will be shown.")
    fmt.println("To look around using first person camera controls, press Space. (Press Space again to switch back)")

    defer cleanup_screen_resources(device)

    linear_sampler := sdl.CreateGPUSampler(device, {
        min_filter = .LINEAR,
        mag_filter = .LINEAR,
        mipmap_mode = .LINEAR,
        address_mode_u = .REPEAT,
        address_mode_v = .REPEAT,
        address_mode_w = .REPEAT,
        max_lod = 1000,
    })
    defer sdl.ReleaseGPUSampler(device, linear_sampler)

    window_size: [2]i32
    max_delta_time: f32 = 1.0 / 10.0  // 10fps
    now_ts := sdl.GetPerformanceCounter()
    ts_freq := sdl.GetPerformanceFrequency()
    for
    {
        proceed := handle_window_events(window)
        if !proceed do break

        if !is_window_valid(window)
        {
            sdl.Delay(16)  // Delay a bit.
            continue
        }

        cmd_buf := sdl.AcquireGPUCommandBuffer(device)
        swapchain: ^sdl.GPUTexture
        ok := sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swapchain, nil, nil)
        ensure(ok)

        // Sometimes the swapchain is NULL even though the function finished successfully (ok = true).
        if swapchain == nil
        {
            sdl.Delay(16)  // Delay a bit.
            ok := sdl.SubmitGPUCommandBuffer(cmd_buf)
            continue
        }

        // Compute delta time
        last_ts := now_ts
        now_ts = sdl.GetPerformanceCounter()
        DELTA_TIME = min(max_delta_time, f32(f64((now_ts - last_ts)*1000) / f64(ts_freq)) / 1000.0)

        old_size  := window_size
        sdl.GetWindowSize(window, &window_size.x, &window_size.y)
        if old_size != window_size {
            rebuild_screen_resources(device, window_size)
        }

        // Main pass
        {
            color_target := sdl.GPUColorTargetInfo {
                texture = MAIN_TARGET_TEXTURE,
                clear_color = { BACKGROUND_COLOR.x, BACKGROUND_COLOR.y, BACKGROUND_COLOR.z, 1.0 },
                load_op = .CLEAR,
                store_op = .STORE,
            }
            depth_target := sdl.GPUDepthStencilTargetInfo {
                texture = MAIN_DEPTH_TEXTURE,
                clear_depth = 1.0,
                load_op = .CLEAR,
                store_op = .STORE,
                stencil_load_op = .DONT_CARE,
                stencil_store_op = .DONT_CARE,
            }

            main_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, &depth_target)
            defer sdl.EndGPURenderPass(main_pass)

            lightmap_screen_percent: f32 = 0.5
            render_screen_percent := 1.0 - lightmap_screen_percent
            render_screen_size := [2]f32 { auto_cast (f32(window_size.x) * render_screen_percent), auto_cast window_size.y }
            lightmap_screen_size := [2]f32 { auto_cast (f32(window_size.x) * lightmap_screen_percent), auto_cast window_size.y } + 1.0

            render_viewport_aspect_ratio := render_screen_size.x / render_screen_size.y

            render_params := lm.Scene_Render_Params {
                depth_only = false,
                render_shadowmap = true,
                viewport_offset = { 0, 0 },
                viewport_size = { auto_cast render_screen_size.x, auto_cast render_screen_size.y },
                world_to_view = compute_world_to_view(),
                view_to_proj = linalg.matrix4_perspective_f32(math.RAD_PER_DEG * 59.0, render_viewport_aspect_ratio, 0.1, 1000.0, false),
                pass = main_pass,
            }
            render_scene(cmd_buf, render_params, lm_tex, linear_sampler, pipelines, mesh, skysphere, sky_tex)
            show_texture(cmd_buf, main_pass, lm_tex, linear_sampler, pipelines, { 1, 0 } * render_screen_size, lightmap_screen_size)
        }

        // Tonemap pass
        {
            color_target := sdl.GPUColorTargetInfo {
                texture = swapchain,
                clear_color = { 0.0, 0.0, 0.0, 1.0 },
                load_op = .DONT_CARE,
                store_op = .STORE,
                cycle = false
            }
            pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, nil)
            defer sdl.EndGPURenderPass(pass)

            tex_binding := sdl.GPUTextureSamplerBinding {
                texture = MAIN_TARGET_TEXTURE,
                sampler = linear_sampler,
            }
            sdl.BindGPUFragmentSamplers(pass, 0, &tex_binding, 1)
            sdl.BindGPUGraphicsPipeline(pass, pipelines.tonemap)
            sdl.DrawGPUPrimitives(
                pass,
                num_vertices   = 6,
                num_instances  = 1,
                first_vertex   = 0,
                first_instance = 0
            )

        }

        ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
        ensure(ok)
        ok = sdl.WaitForGPUSwapchain(device, window)
        ensure(ok)
    }
}

render_scene :: proc(cmd_buf: ^sdl.GPUCommandBuffer, params: lm.Scene_Render_Params, lm_tex: ^sdl.GPUTexture, sampler: ^sdl.GPUSampler, pipelines: Pipelines, mesh: Mesh_GPU, skysphere: Mesh_GPU, sky_tex: ^sdl.GPUTexture)
{
    assert(params.pass != nil)

    // NOTE: There is a "depth_only" member in Scene_Render_Params.
    // In this demo we don't use it for terseness but keep in mind
    // that to reduce baking time a depth-only pass could be used
    // since the lightmap would be black anyway.
    // (This is not true for environment map and any other emissive surface,
    // which are required to be used even if "depth_only" were to be true).
    // There is also a "render_shadowmap" member; if you have shadowmaps those
    // can be rendered at this time (low-res should be fine here).

    sdl.SetGPUViewport(params.pass, {
        x = auto_cast params.viewport_offset.x,
        y = auto_cast params.viewport_offset.y,
        w = auto_cast params.viewport_size.x,
        h = auto_cast params.viewport_size.y,
        min_depth = 0.0,
        max_depth = 1.0
    })

    sdl.SetGPUScissor(params.pass, {
        x = auto_cast params.viewport_offset.x,
        y = auto_cast params.viewport_offset.y,
        w = auto_cast params.viewport_size.x,
        h = auto_cast params.viewport_size.y,
    })

    // Render mesh
    for i in 0..<1
    {
        Uniforms :: struct
        {
            model_to_world: matrix[4, 4]f32,
            model_to_world_normal: matrix[4, 4]f32,
            world_to_proj: matrix[4, 4]f32
        }
        uniforms := Uniforms {
            model_to_world = 1,  // (identity)
            model_to_world_normal = 1,
            world_to_proj = params.view_to_proj * params.world_to_view,
        }
        sdl.PushGPUVertexUniformData(cmd_buf, 0, &uniforms, size_of(Uniforms))

        sdl.BindGPUGraphicsPipeline(params.pass, pipelines.lit)

        lm_tex_binding := sdl.GPUTextureSamplerBinding {
            texture = lm_tex,
            sampler = sampler,
        }
        sdl.BindGPUFragmentSamplers(params.pass, 0, &lm_tex_binding, 1)

        vertex_binding := sdl.GPUBufferBinding {
            buffer = mesh.verts,
            offset = 0
        }
        index_binding := sdl.GPUBufferBinding {
            buffer = mesh.indices,
            offset = 0
        }

        tex_binding := sdl.GPUTextureSamplerBinding {
            texture = lm_tex,
            sampler = sampler,
        }
        sdl.BindGPUFragmentSamplers(params.pass, 0, &tex_binding, 1)

        sdl.BindGPUVertexBuffers(params.pass, 0, &vertex_binding, 1)
        sdl.BindGPUIndexBuffer(params.pass, index_binding, ._32BIT)
        sdl.DrawGPUIndexedPrimitives(
            params.pass,
            num_indices    = auto_cast len(MESH_INDICES),
            num_instances  = 1,
            first_index    = 0,
            vertex_offset  = 0,
            first_instance = 0
        )
    }

    // Render sky
    {
        vertex_binding := sdl.GPUBufferBinding {
            buffer = skysphere.verts,
            offset = 0
        }
        index_binding := sdl.GPUBufferBinding {
            buffer = skysphere.indices,
            offset = 0
        }

        Uniforms :: struct
        {
            world_to_view: matrix[4, 4]f32,
            view_to_proj: matrix[4, 4]f32,
        }
        uniforms := Uniforms {
            world_to_view = params.world_to_view,
            view_to_proj = params.view_to_proj,
        }
        sdl.PushGPUVertexUniformData(cmd_buf, 0, &uniforms, size_of(Uniforms))
        sdl.BindGPUGraphicsPipeline(params.pass, pipelines.sky)

        lm_tex_binding := sdl.GPUTextureSamplerBinding {
            texture = sky_tex,
            sampler = sampler,
        }
        sdl.BindGPUFragmentSamplers(params.pass, 0, &lm_tex_binding, 1)

        sdl.BindGPUVertexBuffers(params.pass, 0, &vertex_binding, 1)
        sdl.BindGPUIndexBuffer(params.pass, index_binding, ._32BIT)
        sdl.DrawGPUIndexedPrimitives(
            params.pass,
            num_indices    = auto_cast len(SKYSPHERE_INDICES),
            num_instances  = 1,
            first_index    = 0,
            vertex_offset  = 0,
            first_instance = 0
        )
    }
}

show_texture :: proc(cmd_buf: ^sdl.GPUCommandBuffer, pass: ^sdl.GPURenderPass, texture: ^sdl.GPUTexture, sampler: ^sdl.GPUSampler, pipelines: Pipelines, viewport_offset: [2]f32, viewport_size: [2]f32)
{
    sdl.SetGPUViewport(pass, {
        x = auto_cast viewport_offset.x,
        y = auto_cast viewport_offset.y,
        w = auto_cast viewport_size.x,
        h = auto_cast viewport_size.y,
        min_depth = 0.0,
        max_depth = 1.0
    })

    sdl.SetGPUScissor(pass, {
        x = auto_cast viewport_offset.x,
        y = auto_cast viewport_offset.y,
        w = auto_cast viewport_size.x,
        h = auto_cast viewport_size.y,
    })

    // Render mesh
    sdl.BindGPUGraphicsPipeline(pass, pipelines.fullscreen_sample_tex)

    lm_tex_binding := sdl.GPUTextureSamplerBinding {
        texture = texture,
        sampler = sampler,
    }
    sdl.BindGPUFragmentSamplers(pass, 0, &lm_tex_binding, 1)

    sdl.DrawGPUPrimitives(
        pass,
        num_vertices   = 6,
        num_instances  = 1,
        first_vertex   = 0,
        first_instance = 0
    )
}

Vertex :: struct
{
    pos: [3]f32,
    normal: [3]f32,
    lm_uv: [2]f32,
}

MESH_VERTS   := #load("resources/mesh_verts", []Vertex)
MESH_INDICES := #load("resources/mesh_indices", []u32)

Mesh_GPU :: struct
{
    verts: ^sdl.GPUBuffer,
    indices: ^sdl.GPUBuffer,
}

Pipelines :: struct
{
    lit: ^sdl.GPUGraphicsPipeline,
    fullscreen_sample_tex: ^sdl.GPUGraphicsPipeline,
    sky: ^sdl.GPUGraphicsPipeline,
    tonemap: ^sdl.GPUGraphicsPipeline,
}

//////////////////////////////////////////////////////////////////////////
// Helper functions. These can be ignored if you're looking for how to
// integrate this library in your codebase, as it's mostly boilerplate.

MAIN_DEPTH_TEXTURE: ^sdl.GPUTexture
MAIN_TARGET_TEXTURE: ^sdl.GPUTexture
MAIN_TARGET_FORMAT: sdl.GPUTextureFormat

rebuild_screen_resources :: proc(device: ^sdl.GPUDevice, new_size: [2]i32)
{
    assert(new_size.x > 0 && new_size.y > 0)

    sdl.ReleaseGPUTexture(device, MAIN_TARGET_TEXTURE)
    MAIN_TARGET_TEXTURE = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = LIGHTMAP_FORMAT,
        width = auto_cast new_size.x,
        height = auto_cast new_size.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = { .SAMPLER, .COLOR_TARGET },
        sample_count = ._1,
    })

    sdl.ReleaseGPUTexture(device, MAIN_DEPTH_TEXTURE)
    MAIN_DEPTH_TEXTURE = sdl.CreateGPUTexture(device, {
        type = .D2,
        format = get_depth_format(device),
        width = auto_cast new_size.x,
        height = auto_cast new_size.y,
        layer_count_or_depth = 1,
        num_levels = 1,
        usage = { .DEPTH_STENCIL_TARGET },
        sample_count = ._1,
    })
}

cleanup_screen_resources :: proc(device: ^sdl.GPUDevice)
{
    sdl.ReleaseGPUTexture(device, MAIN_DEPTH_TEXTURE)
    sdl.ReleaseGPUTexture(device, MAIN_TARGET_TEXTURE)
}

upload_mesh :: proc(device: ^sdl.GPUDevice, mesh_verts: []Vertex, mesh_indices: []u32) -> Mesh_GPU
{
    mesh: Mesh_GPU

    vert_buf_size: u32  = auto_cast len(mesh_verts) * size_of(Vertex)
    index_buf_size: u32 = auto_cast len(mesh_indices) * size_of(u32)

    mesh.verts = sdl.CreateGPUBuffer(device, {
        usage = { .VERTEX },
        size = vert_buf_size,
        props = {}
    })

    mesh.indices = sdl.CreateGPUBuffer(device, {
        usage = { .INDEX },
        size = index_buf_size,
        props = {}
    })

    vert_transfer_buf := sdl.CreateGPUTransferBuffer(device, {
        usage = .UPLOAD,
        size = vert_buf_size,
    })
    defer sdl.ReleaseGPUTransferBuffer(device, vert_transfer_buf)

    index_transfer_buf := sdl.CreateGPUTransferBuffer(device, {
        usage = .UPLOAD,
        size = index_buf_size,
    })
    defer sdl.ReleaseGPUTransferBuffer(device, index_transfer_buf)

    vert_ptr := sdl.MapGPUTransferBuffer(device, vert_transfer_buf, false)
    intr.mem_copy(vert_ptr, raw_data(mesh_verts), vert_buf_size)
    sdl.UnmapGPUTransferBuffer(device, vert_transfer_buf)

    index_ptr := sdl.MapGPUTransferBuffer(device, index_transfer_buf, false)
    intr.mem_copy(index_ptr, raw_data(mesh_indices), index_buf_size)
    sdl.UnmapGPUTransferBuffer(device, index_transfer_buf)

    // Upload transfer data
    {
        cmd_buf := sdl.AcquireGPUCommandBuffer(device)
        pass := sdl.BeginGPUCopyPass(cmd_buf)

        sdl.UploadToGPUBuffer(
            pass,
            source = {
                transfer_buffer = vert_transfer_buf,
                offset = 0,
            },
            destination = {
                buffer = mesh.verts,
                offset = 0,
                size = vert_buf_size,
            },
            cycle = false
        )

        sdl.UploadToGPUBuffer(
            pass,
            source = {
                transfer_buffer = index_transfer_buf,
                offset = 0,
            },
            destination = {
                buffer = mesh.indices,
                offset = 0,
                size = index_buf_size,
            },
            cycle = false
        )

        sdl.EndGPUCopyPass(pass)
        ok := sdl.SubmitGPUCommandBuffer(cmd_buf)
        assert(ok)
    }

    return mesh
}

cleanup_mesh :: proc(device: ^sdl.GPUDevice, mesh: ^Mesh_GPU)
{
    sdl.ReleaseGPUBuffer(device, mesh.verts)
    sdl.ReleaseGPUBuffer(device, mesh.indices)
    mesh^ = {}
}

SKYSPHERE_VERTS   := #load("resources/skysphere_verts", []Vertex)
SKYSPHERE_INDICES := #load("resources/skysphere_indices", []u32)
SKY_TEXTURE := #load("resources/sky.hdr")

upload_sky_resources :: proc(device: ^sdl.GPUDevice) -> (Mesh_GPU, ^sdl.GPUTexture)
{
    skysphere_mesh := upload_mesh(device, SKYSPHERE_VERTS, SKYSPHERE_INDICES)

    // Upload environment map texture
    sky_tex: ^sdl.GPUTexture
    {
        width, height, num_channels: c.int
        loaded_img := stbi.loadf_from_memory(raw_data(SKY_TEXTURE), auto_cast len(SKY_TEXTURE), &width, &height, &num_channels, 4)
        defer stbi.image_free(loaded_img)

        // Convert the image to f16.
        converted_img := make([][4]f16, width * height)
        defer delete(converted_img)

        for i in 0..<width * height
        {
            pixel: [4]f32
            pixel[0] = loaded_img[i * 4 + 0]
            pixel[1] = loaded_img[i * 4 + 1]
            pixel[2] = loaded_img[i * 4 + 2]
            pixel[3] = loaded_img[i * 4 + 3]
            converted_img[i] = { auto_cast pixel[0], auto_cast pixel[1], auto_cast pixel[2], auto_cast pixel[3] }
        }

        num_mip_levels := 2

        // Upload to GPU
        sky_tex = sdl.CreateGPUTexture(device, {
            type = .D2,
            format = .R16G16B16A16_FLOAT,
            width = auto_cast width,
            height = auto_cast height,
            layer_count_or_depth = 1,
            num_levels = auto_cast max(num_mip_levels, 1),
            usage = { .SAMPLER, .COLOR_TARGET },  // COLOR_TARGET is needed to generate mipmaps.
        })

        transfer_buf := sdl.CreateGPUTransferBuffer(device, {
            usage = .UPLOAD,
            size  = auto_cast len(converted_img) * size_of(converted_img[0])
        })
        defer sdl.ReleaseGPUTransferBuffer(device, transfer_buf)

        transfer_dst := sdl.MapGPUTransferBuffer(device, transfer_buf, false)
        intr.mem_copy(transfer_dst, raw_data(converted_img), len(converted_img) * size_of(converted_img[0]))
        sdl.UnmapGPUTransferBuffer(device, transfer_buf)

        cmd_buf := sdl.AcquireGPUCommandBuffer(device)
        pass := sdl.BeginGPUCopyPass(cmd_buf)
        sdl.UploadToGPUTexture(
            pass,
            source = {
                transfer_buffer = transfer_buf,
                offset = 0,
            },
            destination = {
                texture = sky_tex,
                w = auto_cast width,
                h = auto_cast height,
                d = 1,
            },
            cycle = false
        )
        sdl.EndGPUCopyPass(pass)

        if num_mip_levels > 1 {
            sdl.GenerateMipmapsForGPUTexture(cmd_buf, sky_tex)
        }

        ok_s := sdl.SubmitGPUCommandBuffer(cmd_buf)
        assert(ok_s)
    }

    return skysphere_mesh, sky_tex
}

cleanup_sky_resources :: proc(device: ^sdl.GPUDevice, mesh: ^Mesh_GPU, tex: ^^sdl.GPUTexture)
{
    cleanup_mesh(device, mesh)
    sdl.ReleaseGPUTexture(device, tex^)
    tex^ = nil
}

hemi_reduce_spv          :: #load("resources/hemisphere_reduce.comp.spv")
hemi_weighted_reduce_spv :: #load("resources/hemisphere_weighted_reduce.comp.spv")
hemi_reduce_msl          :: #load("resources/hemisphere_reduce.comp.msl")
hemi_weighted_reduce_msl :: #load("resources/hemisphere_weighted_reduce.comp.msl")

make_lm_shaders :: proc(device: ^sdl.GPUDevice) -> lm.Shaders
{
    supported_shader_formats := sdl.GetGPUShaderFormats(device)
    use_spv := .SPIRV in supported_shader_formats
    decided_format: sdl.GPUShaderFormatFlag = .SPIRV if use_spv else .MSL
    assert(use_spv || .MSL in supported_shader_formats)

    // Shaders taken from this repository.
    return lm.Shaders {
        hemi_reduce = sdl.CreateGPUComputePipeline(device, {
            code_size = len(hemi_reduce_spv) if use_spv else len(hemi_reduce_msl),
            code = raw_data(hemi_reduce_spv) if use_spv else raw_data(hemi_reduce_msl),
            format = { decided_format },
            entrypoint = "main",
            num_samplers = 0,
            num_readonly_storage_textures = 1,
            num_readonly_storage_buffers = 0,
            num_readwrite_storage_textures = 1,
            num_readwrite_storage_buffers = 0,
            num_uniform_buffers = 1,
            threadcount_x = 8,
            threadcount_y = 8,
            threadcount_z = 1,
        }),
        hemi_weighted_reduce = sdl.CreateGPUComputePipeline(device, {
            code_size = len(hemi_weighted_reduce_spv) if use_spv else len(hemi_weighted_reduce_msl),
            code = raw_data(hemi_weighted_reduce_spv) if use_spv else raw_data(hemi_weighted_reduce_msl),
            format = { decided_format },
            entrypoint = "main",
            num_samplers = 0,
            num_readonly_storage_textures = 2,
            num_readonly_storage_buffers = 0,
            num_readwrite_storage_textures = 1,
            num_readwrite_storage_buffers = 0,
            num_uniform_buffers = 1,
            threadcount_x = 8,
            threadcount_y = 8,
            threadcount_z = 1,
        }),
    }
}

// User shaders (specific to this example and not part of the library)
lit_frag_spv             := #load("resources/lit.frag.spv")
model_to_proj_vert_spv   := #load("resources/model_to_proj.vert.spv")
fullscreen_quad_vert_spv := #load("resources/fullscreen_quad.vert.spv")
sample_tex_frag_spv      := #load("resources/sample_tex.frag.spv")
sky_vert_spv             := #load("resources/sky.vert.spv")
sky_frag_spv             := #load("resources/sky.frag.spv")
tonemap_frag_spv         := #load("resources/tonemap.frag.spv")
lit_frag_msl             := #load("resources/lit.frag.msl")
model_to_proj_vert_msl   := #load("resources/model_to_proj.vert.msl")
fullscreen_quad_vert_msl := #load("resources/fullscreen_quad.vert.msl")
sample_tex_frag_msl      := #load("resources/sample_tex.frag.msl")
sky_vert_msl             := #load("resources/sky.vert.msl")
sky_frag_msl             := #load("resources/sky.frag.msl")
tonemap_frag_msl         := #load("resources/tonemap.frag.msl")

make_pipelines :: proc(device: ^sdl.GPUDevice, window: ^sdl.Window) -> Pipelines
{
    supported_shader_formats := sdl.GetGPUShaderFormats(device)
    use_spv := .SPIRV in supported_shader_formats
    decided_format: sdl.GPUShaderFormatFlag = .SPIRV if use_spv else .MSL
    assert(use_spv || .MSL in supported_shader_formats)

    lit_frag := sdl.CreateGPUShader(device, {
        code_size = len(lit_frag_spv) if use_spv else len(lit_frag_msl),
        code = raw_data(lit_frag_spv) if use_spv else raw_data(lit_frag_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .FRAGMENT,
        num_samplers = 1,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 0,
        props = {}
    })
    model_to_proj_vert := sdl.CreateGPUShader(device, {
        code_size = len(model_to_proj_vert_spv) if use_spv else len(model_to_proj_vert_msl),
        code = raw_data(model_to_proj_vert_spv) if use_spv else raw_data(model_to_proj_vert_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .VERTEX,
        num_samplers = 0,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 1,
        props = {}
    })
    fullscreen_quad_vert := sdl.CreateGPUShader(device, {
        code_size = len(fullscreen_quad_vert_spv) if use_spv else len(fullscreen_quad_vert_msl),
        code = raw_data(fullscreen_quad_vert_spv) if use_spv else raw_data(fullscreen_quad_vert_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .VERTEX,
        num_samplers = 0,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 0,
        props = {}
    })
    sample_tex_frag := sdl.CreateGPUShader(device, {
        code_size = len(sample_tex_frag_spv) if use_spv else len(sample_tex_frag_msl),
        code = raw_data(sample_tex_frag_spv) if use_spv else raw_data(sample_tex_frag_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .FRAGMENT,
        num_samplers = 1,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 0,
        props = {}
    })
    sky_vert := sdl.CreateGPUShader(device, {
        code_size = len(sky_vert_spv) if use_spv else len(sky_vert_msl),
        code = raw_data(sky_vert_spv) if use_spv else raw_data(sky_vert_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .VERTEX,
        num_samplers = 0,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 1,
        props = {}
    })
    sky_frag := sdl.CreateGPUShader(device, {
        code_size = len(sky_frag_spv) if use_spv else len(sky_frag_msl),
        code = raw_data(sky_frag_spv) if use_spv else raw_data(sky_frag_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .FRAGMENT,
        num_samplers = 1,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 0,
        props = {}
    })
    tonemap_frag := sdl.CreateGPUShader(device, {
        code_size = len(tonemap_frag_spv) if use_spv else len(tonemap_frag_msl),
        code = raw_data(tonemap_frag_spv) if use_spv else raw_data(tonemap_frag_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .FRAGMENT,
        num_samplers = 1,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 0,
        props = {}
    })
    defer
    {
        sdl.ReleaseGPUShader(device, lit_frag)
        sdl.ReleaseGPUShader(device, model_to_proj_vert)
        sdl.ReleaseGPUShader(device, fullscreen_quad_vert)
        sdl.ReleaseGPUShader(device, sample_tex_frag)
        sdl.ReleaseGPUShader(device, sky_vert)
        sdl.ReleaseGPUShader(device, sky_frag)
        sdl.ReleaseGPUShader(device, tonemap_frag)
    }

    // Vertex layout
    vertex_buffer_descriptions := sdl.GPUVertexBufferDescription {
        slot = 0,
        input_rate = .VERTEX,
        instance_step_rate = 0,
        pitch = size_of(Vertex)
    }

    vertex_attributes := [3]sdl.GPUVertexAttribute {
        {
            buffer_slot = 0,
            format = .FLOAT3,
            location = 0,
            offset = auto_cast offset_of(Vertex, pos)
        },
        {
            buffer_slot = 0,
            format = .FLOAT3,
            location = 1,
            offset = auto_cast offset_of(Vertex, normal)
        },
        {
            buffer_slot = 0,
            format = .FLOAT2,
            location = 2,
            offset = auto_cast offset_of(Vertex, lm_uv)
        }
    }

    static_mesh_layout := sdl.GPUVertexInputState {
        num_vertex_buffers = 1,
        vertex_buffer_descriptions = &vertex_buffer_descriptions,
        num_vertex_attributes = len(vertex_attributes),
        vertex_attributes = auto_cast &vertex_attributes,
    }

    // Pipelines
    target_format := LIGHTMAP_FORMAT
    swapchain_format := sdl.GetGPUSwapchainTextureFormat(device, window)

    pipelines: Pipelines
    using pipelines
    lit = sdl.CreateGPUGraphicsPipeline(device, {
        target_info = sdl.GPUGraphicsPipelineTargetInfo {
            num_color_targets = 1,
            color_target_descriptions = raw_data([]sdl.GPUColorTargetDescription {
                { format = target_format },
            }),
            has_depth_stencil_target = true,
            depth_stencil_format = get_depth_format(device),
        },
        vertex_input_state = static_mesh_layout,
        primitive_type = .TRIANGLELIST,
        rasterizer_state = {
            fill_mode = .FILL,
            cull_mode = .NONE,
            front_face = .COUNTER_CLOCKWISE,
            enable_depth_clip = true,
        },
        multisample_state = {
            sample_count = ._1,
        },
        depth_stencil_state = {
            enable_depth_test = true,
            enable_depth_write = true,
            compare_op = .LESS,
        },
        vertex_shader = model_to_proj_vert,
        fragment_shader = lit_frag,
    })
    fullscreen_sample_tex = sdl.CreateGPUGraphicsPipeline(device, {
        target_info = sdl.GPUGraphicsPipelineTargetInfo {
            num_color_targets = 1,
            color_target_descriptions = raw_data([]sdl.GPUColorTargetDescription {
                { format = target_format },
            }),
            has_depth_stencil_target = true,
            depth_stencil_format = get_depth_format(device),
        },
        // vertex_input_state = static_mesh_layout,
        primitive_type = .TRIANGLELIST,
        rasterizer_state = {
            fill_mode = .FILL,
            cull_mode = .BACK,
            front_face = .COUNTER_CLOCKWISE,
            enable_depth_clip = false,
        },
        multisample_state = {
            sample_count = ._1,
        },
        depth_stencil_state = {
            enable_depth_test = false,
            enable_depth_write = false,
            compare_op = .LESS,
        },
        vertex_shader = fullscreen_quad_vert,
        fragment_shader = sample_tex_frag,
    })
    sky = sdl.CreateGPUGraphicsPipeline(device, {
        target_info = sdl.GPUGraphicsPipelineTargetInfo {
            num_color_targets = 1,
            color_target_descriptions = raw_data([]sdl.GPUColorTargetDescription {
                { format = target_format },
            }),
            has_depth_stencil_target = true,
            depth_stencil_format = get_depth_format(device),
        },
        vertex_input_state = static_mesh_layout,
        primitive_type = .TRIANGLELIST,
        rasterizer_state = {
            fill_mode = .FILL,
            cull_mode = .FRONT,
            front_face = .COUNTER_CLOCKWISE
        },
        multisample_state = {
            sample_count = ._1,
        },
        depth_stencil_state = {
            enable_depth_test = true,
            enable_depth_write = true,
            compare_op = .LESS_OR_EQUAL,
        },
        vertex_shader = sky_vert,
        fragment_shader = sky_frag,
    })
    tonemap = sdl.CreateGPUGraphicsPipeline(device, {
        target_info = sdl.GPUGraphicsPipelineTargetInfo {
            num_color_targets = 1,
            color_target_descriptions = raw_data([]sdl.GPUColorTargetDescription {
                { format = swapchain_format },
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
        vertex_shader = fullscreen_quad_vert,
        fragment_shader = tonemap_frag,
    })

    return pipelines
}

get_depth_format :: proc(device: ^sdl.GPUDevice) -> sdl.GPUTextureFormat
{
    depth_format: sdl.GPUTextureFormat

    // SDL docs specify that it's guaranteed for at least
    // one of these two to be supported.
    if sdl.GPUTextureSupportsFormat(device, .D32_FLOAT, .D2, { .DEPTH_STENCIL_TARGET }) {
        depth_format = .D32_FLOAT
    } else {
        depth_format = .D24_UNORM
    }

    return depth_format
}

cleanup_pipelines :: proc(device: ^sdl.GPUDevice, pipelines: ^Pipelines)
{
    sdl.ReleaseGPUGraphicsPipeline(device, pipelines.lit)
    sdl.ReleaseGPUGraphicsPipeline(device, pipelines.fullscreen_sample_tex)
    sdl.ReleaseGPUGraphicsPipeline(device, pipelines.sky)
    sdl.ReleaseGPUGraphicsPipeline(device, pipelines.tonemap)
}

MIN_WINDOW_SIZE: [2]i32 = { 100, 100 }

init_sdl :: proc() -> (^sdl.Window, ^sdl.GPUDevice)
{
    ok_i := sdl.Init({ .VIDEO, .EVENTS })
    ensure(ok_i)

    event: sdl.Event
    window_flags :: sdl.WindowFlags {
        .RESIZABLE,
        .HIGH_PIXEL_DENSITY,
        .HIDDEN,
    }
    window := sdl.CreateWindow("Lightmapper Example", 1700, 1024, window_flags)
    ensure(window != nil)

    debug_mode := true
    device := sdl.CreateGPUDevice({ .SPIRV, .MSL }, debug_mode, nil)
    ensure(device != nil)

    sdl.SetWindowMinimumSize(window, MIN_WINDOW_SIZE.x, MIN_WINDOW_SIZE.y)
    ok_c := sdl.ClaimWindowForGPUDevice(device, window)
    ensure(ok_c)

    ok_f := sdl.SetGPUAllowedFramesInFlight(device, 1)
    ensure(ok_f)

    composition := sdl.GPUSwapchainComposition.SDR
    present_mode := sdl.GPUPresentMode.VSYNC

    ok_s := sdl.SetGPUSwapchainParameters(device, window, composition, present_mode)
    ensure(ok_s)

    return window, device
}

quit_sdl :: proc(window: ^sdl.Window, device: ^sdl.GPUDevice)
{
    sdl.ReleaseWindowFromGPUDevice(device, window)
    sdl.DestroyGPUDevice(device)
    sdl.DestroyWindow(window)
    sdl.Quit()
}

handle_window_events :: proc(window: ^sdl.Window) -> (proceed: bool)
{
    // Reset "one-shot" inputs
    for &key in INPUT.keys
    {
        key.pressed = false
        key.released = false
    }
    INPUT.mouse_dx = 0
    INPUT.mouse_dy = 0

    event: sdl.Event
    proceed = true
    for sdl.PollEvent(&event)
    {
        #partial switch event.type
        {
            case .QUIT:
                proceed = false
            case .WINDOW_CLOSE_REQUESTED:
            {
                if event.window.windowID == sdl.GetWindowID(window) {
                    proceed = false
                }
            }
            // Input events
            case .MOUSE_BUTTON_DOWN, .MOUSE_BUTTON_UP:
            {
                event := event.button
                if event.type == .MOUSE_BUTTON_DOWN {
                    if event.button == sdl.BUTTON_RIGHT {
                        INPUT.pressing_right_click = true
                    }
                } else if event.type == .MOUSE_BUTTON_UP {
                    if event.button == sdl.BUTTON_RIGHT {
                        INPUT.pressing_right_click = false
                    }
                }
            }
            case .KEY_DOWN, .KEY_UP:
            {
                event := event.key
                if event.repeat do break

                if event.type == .KEY_DOWN
                {
                    INPUT.keys[event.scancode].pressed = true
                    INPUT.keys[event.scancode].pressing = true
                }
                else
                {
                    INPUT.keys[event.scancode].pressing = false
                    INPUT.keys[event.scancode].released = true
                }
            }
            case .MOUSE_MOTION:
            {
                event := event.motion
                INPUT.mouse_dx += event.xrel
                INPUT.mouse_dy -= event.yrel  // In sdl, up is negative
            }
        }
    }

    return
}

is_window_valid :: proc(window: ^sdl.Window) -> bool
{
    w, h: i32
    sdl.GetWindowSize(window, &w, &h)
    window_flags := sdl.GetWindowFlags(window)

    res := true
    res &= w >= MIN_WINDOW_SIZE.x && h >= MIN_WINDOW_SIZE.y
    res &= !(.MINIMIZED in window_flags)
    res &= !(.HIDDEN in window_flags)
    return res
}

Key_State :: struct
{
    pressed: bool,
    pressing: bool,
    released: bool,
}

Input :: struct
{
    pressing_right_click: bool,
    keys: #sparse[sdl.Scancode]Key_State,

    mouse_dx: f32,  // pixels/dpi (inches), right is positive
    mouse_dy: f32,  // pixels/dpi (inches), up is positive
}

// A lot of these things are global because this is just an example,
// so I don't want to clutter useful things with boilerplate code.

INPUT: Input

Camera_Movement_Mode :: enum
{
    Rotate_Around_Origin = 0,
    First_Person,
}

DELTA_TIME: f32

compute_world_to_view :: proc() -> matrix[4, 4]f32
{
    @(static) cam_mode := Camera_Movement_Mode.Rotate_Around_Origin
    if INPUT.keys[.SPACE].pressed {
        cam_mode = Camera_Movement_Mode((int(cam_mode) + 1) % len(Camera_Movement_Mode))
    }

    switch cam_mode
    {
        case .Rotate_Around_Origin: return rotating_camera_view()
        case .First_Person:         return first_person_camera_view()
    }

    return 1
}

first_person_camera_view :: proc() -> matrix[4, 4]f32
{
    @(static) cam_pos: [3]f32 = { 0, 2.5, -10 }

    @(static) angle: [2]f32

    cam_rot: quaternion128 = 1

    mouse_sensitivity := math.to_radians_f32(0.2)  // Radians per pixel
    mouse: [2]f32
    if INPUT.pressing_right_click
    {
        mouse.x = INPUT.mouse_dx * mouse_sensitivity
        mouse.y = INPUT.mouse_dy * mouse_sensitivity
    }

    angle += mouse

    // Wrap angle.x
    for angle.x < 0 do angle.x += 2*math.PI
    for angle.x > 2*math.PI do angle.x -= 2*math.PI

    angle.y = clamp(angle.y, math.to_radians_f32(-90), math.to_radians_f32(90))
    y_rot := linalg.quaternion_angle_axis(angle.y, [3]f32 { -1, 0, 0 })
    x_rot := linalg.quaternion_angle_axis(angle.x, [3]f32 { 0, 1, 0 })
    cam_rot = x_rot * y_rot

    // Movement
    @(static) cur_vel: [3]f32
    move_speed: f32 : 6.0
    move_speed_fast: f32 : 15.0
    move_accel: f32 : 300.0

    keyboard_dir_xz: [3]f32
    keyboard_dir_y: f32
    if INPUT.pressing_right_click
    {
        keyboard_dir_xz.x = f32(int(INPUT.keys[.D].pressing) - int(INPUT.keys[.A].pressing))
        keyboard_dir_xz.z = f32(int(INPUT.keys[.W].pressing) - int(INPUT.keys[.S].pressing))
        keyboard_dir_y    = f32(int(INPUT.keys[.E].pressing) - int(INPUT.keys[.Q].pressing))

        // It's a "direction" input so its length
        // should be no more than 1
        if linalg.dot(keyboard_dir_xz, keyboard_dir_xz) > 1 {
            keyboard_dir_xz = linalg.normalize(keyboard_dir_xz)
        }

        if abs(keyboard_dir_y) > 1 {
            keyboard_dir_y = math.sign(keyboard_dir_y)
        }
    }

    target_vel := keyboard_dir_xz * move_speed
    target_vel = linalg.quaternion_mul_vector3(cam_rot, target_vel)
    target_vel.y += keyboard_dir_y * move_speed

    cur_vel = approach_linear(cur_vel, target_vel, move_accel * DELTA_TIME)
    cam_pos += cur_vel * DELTA_TIME

    return world_to_view_mat(cam_pos, cam_rot)

    approach_linear :: proc(cur: [3]f32, target: [3]f32, delta: f32) -> [3]f32
    {
        diff := target - cur
        dist := linalg.length(diff)

        if dist <= delta do return target
        return cur + diff / dist * delta
    }
}

rotating_camera_view :: proc() -> matrix[4, 4]f32
{
    @(static) rot_x: f32
    rot_x = math.mod(rot_x + math.RAD_PER_DEG * 25 * DELTA_TIME, math.RAD_PER_DEG * 360)

    rot := linalg.quaternion_angle_axis(rot_x, [3]f32 { 0, 1, 0 })
    pos := linalg.quaternion_mul_vector3(rot, [3]f32 { 0, 2.5, -10 })

    return world_to_view_mat(pos, rot)
}

world_to_view_mat :: proc(cam_pos: [3]f32, cam_rot: quaternion128) -> matrix[4, 4]f32
{
    view_rot := linalg.normalize(linalg.quaternion_inverse(cam_rot))
    view_pos := -cam_pos
    return #force_inline linalg.matrix4_from_quaternion(view_rot) *
           #force_inline linalg.matrix4_translate(view_pos)
}
