// width, height in pixels
@group(0)
@binding(0)
var<uniform> image_size: vec2u;

@group(1)
@binding(0)
var<storage, read_write> pixels: array<u32>;

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) id: vec3u,
) {
    let aspect_ratio = f32(image_size.x) / f32(image_size.y);
    let viewport_size = vec2f(aspect_ratio, 1.0) * 3.5;
    let step = viewport_size / vec2f(image_size);

    let pos = vec2f(id.xy) * step - viewport_size / 2.0;
    var z = pos;
    var i = 254u;

    while !oob(z, viewport_size) && i > 0u {
        i -= 1u;
        z = square(z) + pos;
    }

    pixels[id.y * image_size.x + id.x] = i;
}

fn square(c: vec2f) -> vec2f {
    // (a+bi)(a+bi) = aa - bb + 2abi
    return vec2f(
        c.x * c.x - c.y * c.y,
        2f * c.x * c.y,
    );
}

fn oob(c: vec2f, viewport_size: vec2f) -> bool {
    return abs(c.x) > viewport_size.x / 2f || abs(c.y) > viewport_size.y / 2f;
}
