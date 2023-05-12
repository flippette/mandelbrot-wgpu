@group(0)
@binding(0)
var<storage, read_write> pixels: array<u32>;

// width, height in pixels
@group(0)
@binding(1)
var<storage, read> image_size: array<u32, 2>;

// width, height in units
@group(0)
@binding(2)
var<storage, read> viewport_size: array<f32, 2>;

@compute
@workgroup_size(64, 64, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    let image_size = vec2<u32>(image_size[0], image_size[1]);
    let viewport_size = vec2<f32>(viewport_size[0], viewport_size[1]);

    let step = viewport_size / vec2<f32>(image_size);

    let pos = vec2<f32>(id.xy);
    var z = vec2<f32>(0f);
    var i = 255u;

    while i > 0u && !oob(z, viewport_size) {
        i -= 1u;
        z = square(z) + pos;
    }

    pixels[id.y * image_size.x + id.x] = i;
}

fn square(c: vec2<f32>) -> vec2<f32> {
    // (a+bi)(a+bi) = aa - bb + 2abi
    return vec2<f32>(
        c.x * c.x - c.y * c.y,
        2f * c.x * c.y,
    );
}

fn oob(c: vec2<f32>, viewport_size: vec2<f32>) -> bool {
    return abs(c.x) > viewport_size.x / 2f || abs(c.y) > viewport_size.y / 2f;
}
