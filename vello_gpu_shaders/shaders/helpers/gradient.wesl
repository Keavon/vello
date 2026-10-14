// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared encoded gradient accessors and gradient evaluation helpers.

import package::helpers::extend::{EXTEND_PAD, extend_mode_normalized};
import package::helpers::math::{NEARLY_ZERO_TOLERANCE, TWO_PI};
import package::helpers::paint::{PAINT_TEXTURE_INDEX_MASK, load_encoded_paint_texel};
import package::helpers::texture::flat_index_to_texture_coord;

fn get_gradient_sample_xy(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    paint_and_rect_flag: u32,
    scene_strip: vec2<u32>,
    corner: vec2<f32>,
    width: u32,
    height: u32,
) -> vec2<f32> {
    let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let gradient_texel0 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            0u,
        );
    let gradient_texel1 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            1u,
        );
    let scene_pos = vec2<f32>(
            f32(scene_strip.x) + corner.x * f32(width),
            f32(scene_strip.y) + corner.y * f32(height),
        );
    return apply_gradient_transform(gradient_texel0, gradient_texel1, scene_pos);
}

fn sample_linear_gradient(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    gradient_texture: texture_2d<f32>,
    paint_and_rect_flag: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let gradient_texel0 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            0u,
        );

    // For linear gradient, t-value is just the x coordinate in gradient space
    let t_value = sample_xy.x;
    let gradient_color = evaluate_gradient(
            gradient_texture,
            t_value,
            get_gradient_extend_mode(gradient_texel0),
            get_gradient_start(gradient_texel0),
        );
    return gradient_color;
}

fn sample_radial_gradient(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    gradient_texture: texture_2d<f32>,
    paint_and_rect_flag: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let gradient_texel0 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            0u,
        );
    let gradient_texel2 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            2u,
        );

    // Pencil-of-circles gradients generalize radial ones, so they share this paint type
    if get_radial_is_pencil(gradient_texel2) {
        let t_value = calculate_pencil_gradient(
                sample_xy,
                gradient_texel2,
                load_encoded_paint_texel(encoded_paints_texture, tex_width_bits, paint_tex_idx, 3u),
                load_encoded_paint_texel(encoded_paints_texture, tex_width_bits, paint_tex_idx, 4u),
                load_encoded_paint_texel(encoded_paints_texture, tex_width_bits, paint_tex_idx, 5u),
            );
        return evaluate_gradient(
                gradient_texture,
                t_value,
                get_gradient_extend_mode(gradient_texel0),
                get_gradient_start(gradient_texel0),
            );
    }

    // For radial gradient, calculate distance from center
    let gradient_result = calculate_radial_gradient(sample_xy, gradient_texel2);
    let gradient_color = evaluate_gradient(
            gradient_texture,
            gradient_result.x,
            get_gradient_extend_mode(gradient_texel0),
            get_gradient_start(gradient_texel0),
        );
    return select(vec4<f32>(0.0, 0.0, 0.0, 0.0), gradient_color, gradient_result.y != 0.0);
}

@if(gradient_sweep)
fn sample_sweep_gradient(
    encoded_paints_texture: texture_2d<u32>,
    tex_width_bits: u32,
    gradient_texture: texture_2d<f32>,
    paint_and_rect_flag: u32,
    sample_xy: vec2<f32>,
) -> vec4<f32> {
    let paint_tex_idx = paint_and_rect_flag & PAINT_TEXTURE_INDEX_MASK;
    let gradient_texel0 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            0u,
        );
    let gradient_texel2 = load_encoded_paint_texel(
            encoded_paints_texture,
            tex_width_bits,
            paint_tex_idx,
            2u,
        );

    // Before passing the position to the angle calculation, we bias
    // very small coordinates to 0. Otherwise the sweep gradient's seam
    // may flicker, because the angle calculation uses the coordinates'
    // signs to select a quadrant. For coordinates around 0, slight
    // noise in the coordinate calculation can then land the
    // calculation in different quadrants. That flickering is quite
    // noticeable as the seam is not anti-aliased. The flickering may
    // vary across machines. See
    // <https://github.com/linebender/vello/pull/1352>.
    let grad_pos = select(sample_xy, vec2(0.0), abs(sample_xy) < vec2(NEARLY_ZERO_TOLERANCE));

    // For sweep gradient, calculate angle from center using fast polynomial approximation
    let unit_angle = xy_to_unit_angle(grad_pos.x, grad_pos.y);
    // Convert unit angle [0, 1) to radians [0, 2π)
    let angle = unit_angle * TWO_PI;
    let t_value = (angle - get_sweep_start_angle(gradient_texel2))
        * get_sweep_inv_angle_delta(gradient_texel2);
    let gradient_color = evaluate_gradient(
            gradient_texture,
            t_value,
            get_gradient_extend_mode(gradient_texel0),
            get_gradient_start(gradient_texel0),
        );
    return gradient_color;
}

const RADIAL_KIND_WELL_BEHAVED: u32 = 0u;
const RADIAL_KIND_STRIP: u32 = 1u;
const RADIAL_KIND_FOCAL_ON_CIRCLE: u32 = 2u;
const RADIAL_KIND_FOCAL: u32 = 3u;

// Evaluate the gradient's color at `t_value` from the record of its stops at `gradient_start`.
//
// The record's first texel holds the stop count and whether the colors are premultiplied, then
// come the offsets four to a texel, then one texel per color. Where several stops share an
// offset, the last of them holds from that offset onward, and padding extends the first and last
// stops' colors beyond the ends, as SVG and CSS specify.
fn evaluate_gradient(
    gradient_texture: texture_2d<f32>,
    t_value: f32,
    extend_mode: u32,
    gradient_start: u32,
) -> vec4<f32> {
    let gradient_tex_width = textureDimensions(gradient_texture).x;
    let header = textureLoad(gradient_texture, flat_index_to_texture_coord(gradient_start, gradient_tex_width), 0);
    let stop_count = u32(header.x);
    let premultiplied = header.y != 0.0;

    // Padding needs no folding, since the search below extends the end stops itself
    let t = select(extend_mode_normalized(t_value, extend_mode), t_value, extend_mode == EXTEND_PAD);

    // Find the first stop past `t`, an upper bound in O(log n) texel reads
    let offsets_start = gradient_start + 1u;
    var low = 0u;
    var high = stop_count;
    while low < high {
        let middle = (low + high) / 2u;
        let offsets = textureLoad(gradient_texture, flat_index_to_texture_coord(offsets_start + middle / 4u, gradient_tex_width), 0);
        if texel_component(offsets, middle % 4u) <= t {
            low = middle + 1u;
        } else {
            high = middle;
        }
    }

    let colors_start = offsets_start + (stop_count + 3u) / 4u;
    let before = min(low, stop_count) - select(1u, 0u, low == 0u);
    let after = min(low, stop_count - 1u);
    let color_before = textureLoad(gradient_texture, flat_index_to_texture_coord(colors_start + before, gradient_tex_width), 0);
    let color_after = textureLoad(gradient_texture, flat_index_to_texture_coord(colors_start + after, gradient_tex_width), 0);

    // Interpolate between the stops around `t`, or hold an end stop's color beyond the ends
    var color = color_before;
    if before != after {
        let offset_before = texel_component(textureLoad(gradient_texture, flat_index_to_texture_coord(offsets_start + before / 4u, gradient_tex_width), 0), before % 4u);
        let offset_after = texel_component(textureLoad(gradient_texture, flat_index_to_texture_coord(offsets_start + after / 4u, gradient_tex_width), 0), after % 4u);
        color = mix(color_before, color_after, (t - offset_before) / (offset_after - offset_before));
    }

    if !premultiplied {
        color = vec4<f32>(color.rgb * color.a, color.a);
    }

    // Keep the color channels within the alpha, which interpolation can overshoot by rounding
    let alpha = clamp(color.a, 0.0, 1.0);
    return vec4<f32>(clamp(color.rgb, vec3(0.0), vec3(alpha)), alpha);
}

// One component of a texel, chosen without dynamic indexing, which older drivers mishandle.
fn texel_component(texel: vec4<f32>, component: u32) -> f32 {
    return select(
        select(texel.x, texel.y, component == 1u),
        select(texel.z, texel.w, component == 3u),
        component >= 2u
    );
}

/// Width of the gradient texture.
fn get_gradient_texture_width(texel0: vec4<u32>) -> u32 {
    return texel0.x & 0x0FFFFFFFu;
}

/// The extend mode for the gradient.
fn get_gradient_extend_mode(texel0: vec4<u32>) -> u32 {
    return (texel0.x >> 30u) & 3u;
}

/// Start coordinate in the flat gradient texture.
fn get_gradient_start(texel0: vec4<u32>) -> u32 {
    return texel0.y;
}

/// 2x2 linear part of the affine transform (columns [a,b] and [c,d]).
fn get_gradient_transform(texel0: vec4<u32>, texel1: vec4<u32>) -> mat2x2<f32> {
    return mat2x2<f32>(
            vec2<f32>(bitcast<f32>(texel0.z), bitcast<f32>(texel0.w)),
            vec2<f32>(bitcast<f32>(texel1.x), bitcast<f32>(texel1.y)),
        );
}

/// Translation part of the affine transform [tx, ty].
fn get_gradient_translate(texel1: vec4<u32>) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(texel1.z), bitcast<f32>(texel1.w));
}

fn apply_gradient_transform(texel0: vec4<u32>, texel1: vec4<u32>, pos: vec2<f32>) -> vec2<f32> {
    return get_gradient_transform(texel0, texel1) * pos + get_gradient_translate(texel1);
}

/// Kind of radial gradient, see `RADIAL_KIND_*`.
fn get_radial_kind(texel2: vec4<u32>) -> u32 {
    return texel2.x & 0x3u;
}

/// Whether to negate the square root of `RADIAL_KIND_FOCAL`.
fn get_radial_negate_root(texel2: vec4<u32>) -> bool {
    return (texel2.x & 0x4u) != 0u;
}

/// Whether the paint is a pencil-of-circles gradient rather than a radial one.
fn get_radial_is_pencil(texel2: vec4<u32>) -> bool {
    return (texel2.x & 0x8u) != 0u;
}

/// `fp0` for focal gradients, `scaled_r0_squared` for strip gradients and 0 otherwise.
fn get_radial_param(texel2: vec4<u32>) -> f32 {
    return bitcast<f32>(texel2.y);
}

/// Scale applied to the computed gradient position.
fn get_radial_scale(texel2: vec4<u32>) -> f32 {
    return bitcast<f32>(texel2.z);
}

/// Bias added to the scaled gradient position.
fn get_radial_bias(texel2: vec4<u32>) -> f32 {
    return bitcast<f32>(texel2.w);
}

/// Starting angle for sweep gradient (in radians).
fn get_sweep_start_angle(texel2: vec4<u32>) -> f32 {
    return bitcast<f32>(texel2.x);
}

/// Inverse of angle delta for sweep gradient.
fn get_sweep_inv_angle_delta(texel2: vec4<u32>) -> f32 {
    return bitcast<f32>(texel2.y);
}

// Fast polynomial approximation for xy_to_unit_angle from Skia
// Returns angle in [0, 1) range representing [0, 2π)
// See: https://github.com/google/skia/blob/30bba741989865c157c7a997a0caebe94921276b/src/opts/SkRasterPipeline_opts.h#L5859
fn xy_to_unit_angle(x: f32, y: f32) -> f32 {
    let xabs = abs(x);
    let yabs = abs(y);
    let slope = min(xabs, yabs) / max(xabs, yabs);
    let s = slope * slope;
    // Use a 7th degree polynomial to approximate atan.
    // This was generated using sollya.gforge.inria.fr.
    // A float optimized polynomial was generated using the following command.
    // P1 = fpminimax((1/(2*Pi))*atan(x),[|1,3,5,7|],[|24...|],[2^(-40),1],relative);
    var phi = slope
        * (0.15912117063999176025390625
            + s
            * (-5.185396969318389892578125e-2
                + s
                * (2.476101927459239959716796875e-2 + s * (-7.0547382347285747528076171875e-3))));
    // Map from first octant to full circle using quadrant information
    // Handle [0°, 90°] range
    phi = select(phi, 0.25 - phi, xabs < yabs);
    // Handle [90°, 180°] range
    phi = select(phi, 0.5 - phi, x < 0.0);
    // Handle [180°, 360°] range
    phi = select(phi, 1.0 - phi, y < 0.0);
    // Handle NaN cases (using property that NaN != NaN)
    phi = select(phi, 0.0, phi != phi);
    return phi;
}

// Calculate a radial gradient; matches vello_cpu implementation.
fn calculate_radial_gradient(grad_pos: vec2<f32>, texel2: vec4<u32>) -> vec2<f32> {
    let x_pos = grad_pos.x;
    let y_pos = grad_pos.y;
    let param = get_radial_param(texel2);

    var t: f32;
    var is_valid = true;

    switch get_radial_kind(texel2) {
        case RADIAL_KIND_WELL_BEHAVED { t = sqrt(x_pos * x_pos + y_pos * y_pos) - x_pos * param; }
        case RADIAL_KIND_STRIP {
            let p1 = param - y_pos * y_pos;
            is_valid = p1 >= 0.0;
            t = x_pos + sqrt(p1);
        }
        case RADIAL_KIND_FOCAL_ON_CIRCLE {
            t = x_pos + y_pos * y_pos / x_pos;
            is_valid = (t >= 0.0) & (x_pos != 0.0);
        }
        case RADIAL_KIND_FOCAL, default {
            let discriminant = x_pos * x_pos - y_pos * y_pos;
            let root = sqrt(discriminant);
            t = select(root, -root, get_radial_negate_root(texel2)) - x_pos * param;
            is_valid = (discriminant >= 0.0) & (t >= 0.0);
        }
    }

    t = get_radial_scale(texel2) * t + get_radial_bias(texel2);

    return vec2<f32>(t, select(0.0, 1.0, is_valid));
}

/// Stands in for an infinite gradient position, on contours that cross the gradient line at infinity.
const PENCIL_UNBOUNDED_T: f32 = 1e6;

/// A root as a (numerator, denominator) pair, scaled so the larger part has a magnitude of 1.
fn normalized_pencil_root(numerator: f32, denominator: f32) -> vec2<f32> {
    let root = vec2<f32>(numerator, denominator);
    let scale = max(abs(numerator), abs(denominator));
    return select(root / scale, root, scale == 0.0);
}

// Calculate a pencil-of-circles gradient; matches `PencilGeometry::t` in vello_common.
//
// The member of the pencil through the point keeps the ratio of the two end contours' values, so
// it crosses the gradient line at the roots of a quadratic. The root that belongs to the point is
// told apart by its side of the radical axis, or by the arc it lies on, and is then mapped linearly
// to the gradient position.
fn calculate_pencil_gradient(
    grad_pos: vec2<f32>,
    texel2: vec4<u32>,
    texel3: vec4<u32>,
    texel4: vec4<u32>,
    texel5: vec4<u32>,
) -> f32 {
    // `texel2.x` holds the flag that marks a pencil gradient, where radial gradients keep their kind
    let start = bitcast<vec3<f32>>(texel2.yzw);
    let end = bitcast<vec3<f32>>(texel3.xyz);
    let selection_mode = texel3.w;
    let selection = bitcast<vec3<f32>>(texel4.xyz);
    let wrap_reference_x = bitcast<f32>(texel4.w);
    let wrap_band_side = bitcast<f32>(texel5.x);
    let wrap_t = bitcast<f32>(texel5.y);
    let t_scale = bitcast<f32>(texel5.z);
    let t_offset = bitcast<f32>(texel5.w);

    // Normalized so the products below stay in range far from the origin
    let x = grad_pos.x;
    let length_squared = dot(grad_pos, grad_pos);
    let start_raw = start.x * length_squared + start.y * x + start.z;
    let end_raw = end.x * length_squared + end.y * x + end.z;
    let scale = max(abs(start_raw), abs(end_raw));
    if scale == 0.0 {
        return 0.0;
    }
    let start_value = start_raw / scale;
    let end_value = end_raw / scale;

    let a = end_value * start.x - start_value * end.x;
    let b = end_value * start.y - start_value * end.y;
    let c = end_value * start.z - start_value * end.z;
    let discriminant = max(b * b - 4.0 * a * c, 0.0);
    let q = -0.5 * (b + select(-1.0, 1.0, b >= 0.0) * sqrt(discriminant));
    let first = normalized_pencil_root(q, a);
    let second = normalized_pencil_root(c, q);

    var crossing: vec2<f32>;
    if selection_mode == 0u {
        // A point on the radical axis is on the member that is a line, the finite root
        let radical_x = selection.x;
        let first_side = (first.x - radical_x * first.y) * first.y * (x - radical_x);
        let second_side = (second.x - radical_x * second.y) * second.y * (x - radical_x);
        let first_wins = select(first_side > second_side, abs(first.y) >= abs(second.y), first_side == second_side);
        crossing = select(second, first, first_wins);
    } else {
        let first_arc = selection.x * first.x * first.x + selection.y * first.x * first.y + selection.z * first.y * first.y;
        let second_arc = selection.x * second.x * second.x + selection.y * second.x * second.y + selection.z * second.y * second.y;
        crossing = select(second, first, first_arc <= second_arc);
    }

    if wrap_band_side != 0.0 && (crossing.x - wrap_reference_x * crossing.y) * crossing.y * wrap_band_side < 0.0 {
        return wrap_t;
    }
    if crossing.y == 0.0 {
        return sign(crossing.x * t_scale) * PENCIL_UNBOUNDED_T;
    }
    return crossing.x / crossing.y * t_scale + t_offset;
}
