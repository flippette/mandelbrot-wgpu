use bytemuck::{Pod, Zeroable};
use eyre::{eyre, Result};
use std::{fs, io::Write, iter, mem, sync::mpsc};
use tracing::{debug, info, Level};
use tracing_subscriber::EnvFilter;
use wgpu::util::DeviceExt;

const IMAGE_WIDTH: u32 = 4000;
const IMAGE_HEIGHT: u32 = 3000;

fn main() -> Result<()> {
    color_eyre::install()?;
    tracing_subscriber::fmt()
        .compact()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(Level::WARN.into())
                .from_env()?,
        )
        .try_init()
        .map_err(|e| eyre!("failed to initialize logging: {e:?}"))?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    info!("created wgpu instance!");

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .ok_or_else(|| eyre!("failed to request adapter!"))?;
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))?;
    info!("got wgpu device and queue!");

    let workgroup_width =
        (device.limits().max_compute_invocations_per_workgroup as f32).sqrt() as u32;
    assert!(
        workgroup_width * workgroup_width <= device.limits().max_compute_invocations_per_workgroup
    );
    info!(
        "using workgroups of size {}!",
        workgroup_width * workgroup_width
    );

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader module"),
        source: wgpu::ShaderSource::Wgsl(fs::read_to_string("mandelbrot.wgsl")?.into()),
    });
    info!("created shader module!");

    let image_size_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image size bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let storage_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("storage bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    info!("created bind group layouts!");

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute pipeline layout"),
        bind_group_layouts: &[&image_size_bind_group_layout, &storage_bind_group_layout],
        push_constant_ranges: &[],
    });
    info!("created compute pipeline layout!");

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    });
    info!("created compute pipeline!");

    let image_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("image size buffer"),
        contents: bytemuck::cast_slice(&[Vec2U {
            x: IMAGE_WIDTH,
            y: IMAGE_HEIGHT,
        }]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging buffer"),
        size: IMAGE_WIDTH as u64 * IMAGE_HEIGHT as u64 * mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("storage buffer"),
        size: staging_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    info!("created buffers!");

    let image_size_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("image size bind group"),
        layout: &image_size_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: image_size_buffer.as_entire_binding(),
        }],
    });
    let storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("storage bind group"),
        layout: &storage_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    info!("created bind groups!");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("command encoder"),
    });
    info!("created command encoder!");

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compute pass"),
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &image_size_bind_group, &[]);
    pass.set_bind_group(1, &storage_bind_group, &[]);
    pass.insert_debug_marker("compute pass");
    pass.dispatch_workgroups(
        IMAGE_WIDTH / workgroup_width,
        IMAGE_HEIGHT / workgroup_width,
        1,
    );
    drop(pass);
    info!("recorded compute pass!");

    encoder.copy_buffer_to_buffer(
        &storage_buffer,
        0,
        &staging_buffer,
        0,
        staging_buffer.size(),
    );
    info!("queued storage->staging buffer copy!");

    queue.submit(iter::once(encoder.finish()));
    info!("submitted work to device queue!");

    let buffer_slice = staging_buffer.slice(..);

    let (tx, rx) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
    info!("mapped staging buffer for reading!");

    device.poll(wgpu::MaintainBase::Wait);

    // let Ok(Ok(result)) = rx.recv() {
    //     info!("got result!");
    //     let data = buffer_slice.get_mapped_range();
    //     data.iter().step_by(4).copied().collect::<Vec<_>>()
    // };

    let result: Vec<u8> = rx.recv()?.map(|_| {
        buffer_slice
            .get_mapped_range()
            .iter()
            .step_by(4)
            .copied()
            .collect::<Vec<_>>()
    })?;

    debug!("{:?}", &result[0..32]);

    staging_buffer.unmap();

    let mut file = fs::File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open("image.pgm")?;
    let header = format!("P5\n{} {}\n255\n", IMAGE_WIDTH, IMAGE_HEIGHT);
    file.write_all(header.as_bytes())?;
    file.write_all(&result)?;
    info!("wrote image data!");

    Ok(())
}

// for creating WGSL uniforms
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vec2U {
    x: u32,
    y: u32,
}
