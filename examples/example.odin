
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

import sdl "vendor:sdl3"
import lm "../"

BACKGROUND_COLOR :: [3]f32 { 1, 0.2, 0.2 }

main :: proc()
{
    window, device := init_sdl()
    ensure(window != nil && device != nil)
    defer quit_sdl(window, device)

    ts_freq := sdl.GetPerformanceFrequency()

    mesh := upload_mesh(device)
    defer cleanup_mesh(device, &mesh)

    // Pipelines must be compatible with the lightmapping passes (hemisphere_target_format)
    pipelines := make_pipelines(device)
    defer cleanup_pipelines(device, &pipelines)

    // There are many different strategies for loading shaders
    // in SDL GPU, so this library leaves this responsibility
    // to the user. The shaders can be found in this repository.
    // This example will just use precompiled shaders embedded
    // into the executable (in SPIRV and MSL formats, for windows,
    // linux and mac).
    lm_shaders := make_lm_shaders(device)
    defer lm.destroy_shaders(device, lm_shaders)

    _lm_ctx := lm.init(device, lm_shaders, .B8G8R8A8_UNORM, get_depth_format(device), background_color = BACKGROUND_COLOR)
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

    LIGHTMAP_SIZE: [2]int : { 1024, 1024 }

    // The format is fixed to .R16G16B16A16_FLOAT. You're free
    // to convert it to whichever format you prefer. This texture
    // is sampleable so it can be fed back to your renderer,
    // to get multiple bounces (lightmap.tex to access it).
    lightmap := lm.make_lightmap(lm_ctx, LIGHTMAP_SIZE)
    defer lm.delete_lightmap(lm_ctx, lightmap)

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

    // Submit the first frame. Makes it possible to debug lightmap baking.
    when true
    {
        cmd_buf := sdl.AcquireGPUCommandBuffer(device)
        swapchain: ^sdl.GPUTexture
        ok := sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swapchain, nil, nil)
        ensure(ok)
        ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
        ensure(ok)
        ok = sdl.WaitForGPUSwapchain(device, window)
        ensure(ok)
    }

    // Build lightmap.
    {
        fmt.println("Started to bake lightmap...")
        bake_begin_ts := sdl.GetPerformanceCounter()

        NUM_BOUNCES :: 1  // 1 one ambient occlusion only, 2 or more for global illumination.
        for bounce in 0..<NUM_BOUNCES
        {
            lm.set_target_lightmap(lm_ctx, lightmap)
            model_to_world: matrix[4, 4]f32 = 1
            lm.set_target_mesh(lm_ctx, mesh_info, model_to_world)

            for render_params in lm.bake_iterate_begin(lm_ctx)
            {
                defer lm.bake_iterate_end(lm_ctx)

                cmd_buf := render_params.cmd_buf

                render_scene(cmd_buf, render_params, lightmap.tex, linear_sampler, pipelines, mesh)

                progress := lm.bake_progress(lm_ctx)
                //fmt.printfln("Bounce %v, progress: %v", bounce, progress)
            }

            bake_end_ts := sdl.GetPerformanceCounter()
            elapsed := f32(f64((bake_end_ts - bake_begin_ts)*1000) / f64(ts_freq)) / 1000.0

            when ODIN_DEBUG {
                fmt.printfln("Done with lightmap baking! (%.6fs) (DEBUG BUILD)", elapsed)
            } else {
                fmt.printfln("Done with lightmap baking! (%.6fs)", elapsed)
            }
        }
    }

    // Submit the first frame. Makes it possible to debug lightmap baking.
    when true
    {
        cmd_buf = sdl.AcquireGPUCommandBuffer(device)
        ok = sdl.WaitAndAcquireGPUSwapchainTexture(cmd_buf, window, &swapchain, nil, nil)
        ensure(ok)
        ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
        ensure(ok)
        ok = sdl.WaitForGPUSwapchain(device, window)
        ensure(ok)
    }

    sdl.ShowWindow(window)
    view_results(window, device, lightmap.tex, linear_sampler, pipelines, mesh)
}

view_results :: proc(window: ^sdl.Window, device: ^sdl.GPUDevice, lm_tex: ^sdl.GPUTexture, lm_sampler: ^sdl.GPUSampler, pipelines: Pipelines, mesh: Mesh_GPU)
{
    fmt.println("A view of the result will be shown.")
    fmt.println("To look around using first person camera controls, press Space. (Press Space again to go back)")

    defer cleanup_screen_resources(device)

    window_size: [2]i32
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

        old_size  := window_size
        sdl.GetWindowSize(window, &window_size.x, &window_size.y)
        if old_size != window_size {
            rebuild_screen_resources(device, window_size)
        }

        world_to_view := compute_world_to_view()

        color_target := sdl.GPUColorTargetInfo {
            texture = swapchain,
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

        // Main pass
        {
            main_pass := sdl.BeginGPURenderPass(cmd_buf, &color_target, 1, &depth_target)
            defer sdl.EndGPURenderPass(main_pass)

            lightmap_screen_percent: f32 = 0.3
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
            render_scene(cmd_buf, render_params, lm_tex, lm_sampler, pipelines, mesh)
            show_texture(cmd_buf, main_pass, lm_tex, lm_sampler, pipelines, { 1, 0 } * render_screen_size, lightmap_screen_size)
        }

        ok = sdl.SubmitGPUCommandBuffer(cmd_buf)
        ensure(ok)
        ok = sdl.WaitForGPUSwapchain(device, window)
        ensure(ok)
    }
}

render_scene :: proc(cmd_buf: ^sdl.GPUCommandBuffer, params: lm.Scene_Render_Params, lm_tex: ^sdl.GPUTexture, lm_sampler: ^sdl.GPUSampler, pipelines: Pipelines, mesh: Mesh_GPU)
{
    assert(params.pass != nil)

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
        sampler = lm_sampler,
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

    // Render skybox

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
    depth_only: ^sdl.GPUGraphicsPipeline,
    fullscreen_sample_tex: ^sdl.GPUGraphicsPipeline,
}

//////////////////////////////////////////////////////////////////////////
// Helper functions. These can be ignored if you're looking for how to
// integrate this library in your codebase, as it's mostly boilerplate.

MAIN_DEPTH_TEXTURE: ^sdl.GPUTexture
MAIN_TARGET_FORMAT: sdl.GPUTextureFormat

rebuild_screen_resources :: proc(device: ^sdl.GPUDevice, new_size: [2]i32)
{
    assert(new_size.x > 0 && new_size.y > 0)
    if MAIN_DEPTH_TEXTURE != nil do sdl.ReleaseGPUTexture(device, MAIN_DEPTH_TEXTURE)

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
    if MAIN_DEPTH_TEXTURE != nil do sdl.ReleaseGPUTexture(device, MAIN_DEPTH_TEXTURE)
}

upload_mesh :: proc(device: ^sdl.GPUDevice) -> Mesh_GPU
{
    mesh: Mesh_GPU

    vert_buf_size: u32  = auto_cast len(MESH_VERTS) * size_of(Vertex)
    index_buf_size: u32 = auto_cast len(MESH_INDICES) * size_of(u32)

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
    intr.mem_copy(vert_ptr, raw_data(MESH_VERTS), vert_buf_size)
    sdl.UnmapGPUTransferBuffer(device, vert_transfer_buf)

    index_ptr := sdl.MapGPUTransferBuffer(device, index_transfer_buf, false)
    intr.mem_copy(index_ptr, raw_data(MESH_INDICES), index_buf_size)
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

hemi_reduce_spv          :: #load("resources/hemisphere_reduce.comp.spv")
hemi_weighted_reduce_spv :: #load("resources/hemisphere_weighted_reduce.comp.spv")
hemi_reduce_msl          :: #load("resources/hemisphere_reduce.comp.msl")
hemi_weighted_reduce_msl :: #load("resources/hemisphere_weighted_reduce.comp.msl")
blit_lightmap_spv        :: #load("resources/blit_lightmap.frag.spv")
blit_lightmap_msl        :: #load("resources/blit_lightmap.frag.msl")

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
            code_size = len(hemi_reduce_spv) if use_spv else len(hemi_reduce_msl),
            code = raw_data(hemi_reduce_spv) if use_spv else raw_data(hemi_reduce_msl),
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
        fullscreen_quad = sdl.CreateGPUShader(device, {
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
        }),
        blit_lightmap = sdl.CreateGPUShader(device, {
            code_size = len(blit_lightmap_spv) if use_spv else len(blit_lightmap_msl),
            code = raw_data(blit_lightmap_spv) if use_spv else raw_data(blit_lightmap_msl),
            entrypoint = "main",
            format = { decided_format },
            stage = .FRAGMENT,
            num_samplers = 0,
            num_storage_textures = 1,
            num_storage_buffers  = 0,
            num_uniform_buffers = 0,
            props = {}
        }),
    }
}

// User shaders (specific to this example and not part of the library)
lit_frag_spv             := #load("resources/lit.frag.spv")
depth_only_vert_spv      := #load("resources/depth_only.vert.spv")
depth_only_frag_spv      := #load("resources/depth_only.frag.spv")
model_to_proj_vert_spv   := #load("resources/model_to_proj.vert.spv")
fullscreen_quad_vert_spv := #load("resources/fullscreen_quad.vert.spv")
sample_tex_frag_spv      := #load("resources/sample_tex.frag.spv")
lit_frag_msl             := #load("resources/lit.frag.msl")
depth_only_vert_msl      := #load("resources/depth_only.vert.msl")
depth_only_frag_msl      := #load("resources/depth_only.frag.msl")
model_to_proj_vert_msl   := #load("resources/model_to_proj.vert.msl")
fullscreen_quad_vert_msl := #load("resources/fullscreen_quad.vert.msl")
sample_tex_frag_msl      := #load("resources/sample_tex.frag.msl")

make_pipelines :: proc(device: ^sdl.GPUDevice) -> Pipelines
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
    depth_only_vert := sdl.CreateGPUShader(device, {
        code_size = len(depth_only_vert_spv) if use_spv else len(depth_only_vert_msl),
        code = raw_data(depth_only_vert_spv) if use_spv else raw_data(depth_only_vert_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .VERTEX,
        num_samplers = 0,
        num_storage_textures = 0,
        num_storage_buffers  = 0,
        num_uniform_buffers = 1,
        props = {}
    })
    depth_only_frag := sdl.CreateGPUShader(device, {
        code_size = len(depth_only_frag_spv) if use_spv else len(depth_only_frag_msl),
        code = raw_data(depth_only_frag_spv) if use_spv else raw_data(depth_only_frag_msl),
        entrypoint = "main",
        format = { decided_format },
        stage = .FRAGMENT,
        num_samplers = 0,
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
    defer
    {
        sdl.ReleaseGPUShader(device, lit_frag)
        sdl.ReleaseGPUShader(device, depth_only_vert)
        sdl.ReleaseGPUShader(device, depth_only_frag)
        sdl.ReleaseGPUShader(device, model_to_proj_vert)
        sdl.ReleaseGPUShader(device, fullscreen_quad_vert)
        sdl.ReleaseGPUShader(device, sample_tex_frag)
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
    target_format := sdl.GPUTextureFormat.B8G8R8A8_UNORM

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
        vertex_input_state = static_mesh_layout,
        primitive_type = .TRIANGLELIST,
        rasterizer_state = {
            fill_mode = .FILL,
            cull_mode = .BACK,
            front_face = .COUNTER_CLOCKWISE,
            enable_depth_clip = true,
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

    /*
    depth_only = sdl.CreateGPUGraphicsPipeline(device, {

    })
    */

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
    sdl.ReleaseGPUGraphicsPipeline(device, pipelines.depth_only)
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

Input :: struct
{
    pressing_w: bool,
    pressing_a: bool,
    pressing_s: bool,
    pressing_d: bool,
    pressing_q: bool,
    pressing_e: bool,
    pressing_right_click: bool,

    pressed_space: bool,
}

INPUT: Input

Camera_Movement_Mode :: enum
{
    Rotate_Around_Origin = 0,
    First_Person,
}

compute_world_to_view :: proc() -> matrix[4, 4]f32
{
    @(static) cam_mode := Camera_Movement_Mode.Rotate_Around_Origin
    if INPUT.pressed_space {
        cam_mode = Camera_Movement_Mode((int(cam_mode) + 1) % len(Camera_Movement_Mode))
    }

    switch cam_mode
    {
        case .Rotate_Around_Origin: return rotating_camera_view()
        case .First_Person:         return first_person_camera_view()
    }

    return {}
}

first_person_camera_view :: proc() -> matrix[4, 4]f32
{
    @(static) pos: [3]f32
    @(static) rot_x: f32
    @(static) rot_y: f32

    return {}
}

rotating_camera_view :: proc() -> matrix[4, 4]f32
{
    @(static) rot_x: f32
    rot_x = math.mod(rot_x + math.RAD_PER_DEG * 0.3, math.RAD_PER_DEG * 360)

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
