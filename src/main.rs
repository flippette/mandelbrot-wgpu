use eyre::{eyre, Result};
use once_cell::sync::Lazy;
use std::{fs, iter, mem, sync::mpsc};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;
use wgpu::util::DeviceExt;

const IMAGE_WIDTH: u32 = 4000;
const IMAGE_HEIGHT: u32 = 3000;
const VIEWPORT_WIDTH: f32 = 1.0;
const VIEWPORT_HEIGHT: f32 = 0.667;

static WORKGROUP_WIDTH: Lazy<u32> = Lazy::new(|| {
    const WORKGROUP_SIZE: u32 = 64;

    let w = (WORKGROUP_SIZE as f32).sqrt() as u32;
    assert_eq!(w, WORKGROUP_SIZE);

    w
});

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

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader module"),
        source: wgpu::ShaderSource::Wgsl(fs::read_to_string("mandelbrot.wgsl")?.into()),
    });
    info!("created shader module!");

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compute pipeline"),
        layout: None,
        module: &shader_module,
        entry_point: "main",
    });
    info!("created compute pipeline!");

    let image_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("image size buffer"),
        contents: bytemuck::cast_slice(&[IMAGE_WIDTH, IMAGE_HEIGHT]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let viewport_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viewport size buffer"),
        contents: bytemuck::cast_slice(&[VIEWPORT_WIDTH, VIEWPORT_HEIGHT]),
        usage: wgpu::BufferUsages::STORAGE,
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

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: image_size_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: viewport_size_buffer.as_entire_binding(),
            },
        ],
    });
    info!("created bind group!");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("command encoder"),
    });
    info!("created command encoder!");

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("compute pass"),
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.insert_debug_marker("compute pass");
    pass.dispatch_workgroups(
        IMAGE_WIDTH / *WORKGROUP_WIDTH,
        IMAGE_HEIGHT / *WORKGROUP_WIDTH,
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

    let mut result = None;
    if let Ok(Ok(_)) = rx.recv() {
        info!("got result!");
        let data = buffer_slice.get_mapped_range();
        result = Some(bytemuck::cast_slice::<_, u32>(&data).to_vec());
    }

    staging_buffer.unmap();

    let pgm_data = format!(
        "P5\n{} {}\n255\n{}",
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        result
            .map(|pixels| pixels
                .iter()
                .map(|px| char::from_u32(*px).unwrap())
                .collect::<String>())
            .ok_or_else(|| eyre!("failed to compute result!"))?
    );
    info!("created pgm image data!");

    fs::write("image.pgm", pgm_data)?;
    info!("wrote image data!");

    Ok(())
}
