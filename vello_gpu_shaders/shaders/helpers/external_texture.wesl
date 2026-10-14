// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Sampling helpers for externally bound image textures.

import package::helpers::image::{
    IMAGE_QUALITY_HIGH,
    IMAGE_QUALITY_LOW,
    IMAGE_QUALITY_MEDIUM,
    cubic_weights
};

// This is the same as `helpers::image::bilinear_sample`, but for a 2D texture
// instead of an atlas texture array.
fn external_bilinear_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    let image_max = image_offset + image_size - vec2(1.0);
    let clamped_coords = clamp(coords, image_offset, image_max);
    let coord_quad = vec4(floor(clamped_coords), ceil(clamped_coords));
    let coord_frac = fract(coords);
    let a = textureLoad(external_texture, vec2<i32>(coord_quad.xy), 0);
    let b = textureLoad(external_texture, vec2<i32>(coord_quad.xw), 0);
    let c = textureLoad(external_texture, vec2<i32>(coord_quad.zy), 0);
    let d = textureLoad(external_texture, vec2<i32>(coord_quad.zw), 0);
    return mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
}

// This is the same as `helpers::image::bicubic_sample`, but for a 2D texture
// instead of an atlas texture array.
fn external_bicubic_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
) -> vec4<f32> {
    let image_max = image_offset + image_size - vec2(1.0);
    let frac_coords = fract(coords + 0.5);
    let cx = cubic_weights(frac_coords.x);
    let cy = cubic_weights(frac_coords.y);

    let s00 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, -1.5), image_offset, image_max)), 0);
    let s10 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, -1.5), image_offset, image_max)), 0);
    let s20 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, -1.5), image_offset, image_max)), 0);
    let s30 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, -1.5), image_offset, image_max)), 0);

    let s01 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, -0.5), image_offset, image_max)), 0);
    let s11 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, -0.5), image_offset, image_max)), 0);
    let s21 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, -0.5), image_offset, image_max)), 0);
    let s31 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, -0.5), image_offset, image_max)), 0);

    let s02 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, 0.5), image_offset, image_max)), 0);
    let s12 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, 0.5), image_offset, image_max)), 0);
    let s22 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, 0.5), image_offset, image_max)), 0);
    let s32 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, 0.5), image_offset, image_max)), 0);

    let s03 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-1.5, 1.5), image_offset, image_max)), 0);
    let s13 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(-0.5, 1.5), image_offset, image_max)), 0);
    let s23 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(0.5, 1.5), image_offset, image_max)), 0);
    let s33 = textureLoad(external_texture, vec2<i32>(clamp(coords + vec2(1.5, 1.5), image_offset, image_max)), 0);

    let row0 = cx.x * s00 + cx.y * s10 + cx.z * s20 + cx.w * s30;
    let row1 = cx.x * s01 + cx.y * s11 + cx.z * s21 + cx.w * s31;
    let row2 = cx.x * s02 + cx.y * s12 + cx.z * s22 + cx.w * s32;
    let row3 = cx.x * s03 + cx.y * s13 + cx.z * s23 + cx.w * s33;
    let result = cy.x * row0 + cy.y * row1 + cy.z * row2 + cy.w * row3;

    // Clamp alpha first, then clamp premultiplied color channels against it.
    let a = clamp(result.a, 0.0, 1.0);
    return vec4<f32>(clamp(result.rgb, vec3(0.0), vec3(a)), a);
}

// Bilinear sampling of one mipmap level, whose texels each average a block of 2^level texels of the full image along each side.
// `coords` are in the full image's texels.
fn external_mipmap_sample(
    external_texture: texture_2d<f32>,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    level: u32,
) -> vec4<f32> {
    let scale = f32(1u << level);
    // Each level halves the one before it, rounding down, as the texture's levels are sized
    let level_size = max(floor(image_size / scale), vec2(1.0));
    let level_offset = floor(image_offset / scale);
    let level_max = level_offset + level_size - vec2(1.0);

    let level_coords = (coords - image_offset) / scale + level_offset - vec2(0.5);
    let clamped_coords = clamp(level_coords, level_offset, level_max);
    let coord_quad = vec4(floor(clamped_coords), min(ceil(clamped_coords), level_max));
    let coord_frac = fract(clamped_coords);
    let a = textureLoad(external_texture, vec2<i32>(coord_quad.xy), i32(level));
    let b = textureLoad(external_texture, vec2<i32>(coord_quad.xw), i32(level));
    let c = textureLoad(external_texture, vec2<i32>(coord_quad.zy), i32(level));
    let d = textureLoad(external_texture, vec2<i32>(coord_quad.zw), i32(level));
    return mix(mix(a, b, coord_frac.y), mix(c, d, coord_frac.y), coord_frac.x);
}

// Samples an external image. `level` is the base 2 logarithm of the texels the image steps across per pixel. Above zero, the
// smooth qualities blend the two nearest mipmap levels up to `last_level`, so a minified image averages its texels
// rather than skipping over them.
fn sample_external_image(
    external_texture: texture_2d<f32>,
    quality: u32,
    coords: vec2<f32>,
    image_offset: vec2<f32>,
    image_size: vec2<f32>,
    level: f32,
    last_level: f32,
) -> vec4<f32> {
    if quality != IMAGE_QUALITY_LOW && level > 0.0 && last_level > 0.0 {
        let clamped_level = min(level, last_level);
        let lower_level = floor(clamped_level);
        let lower = external_mipmap_sample(external_texture, coords, image_offset, image_size, u32(lower_level));
        if clamped_level == lower_level {
            return lower;
        }
        let upper = external_mipmap_sample(external_texture, coords, image_offset, image_size, u32(lower_level) + 1u);
        return mix(lower, upper, clamped_level - lower_level);
    }

    if quality == IMAGE_QUALITY_HIGH {
        return external_bicubic_sample(external_texture, coords, image_offset, image_size);
    }
    if quality == IMAGE_QUALITY_MEDIUM {
        return external_bilinear_sample(
            external_texture,
            coords - vec2(0.5),
            image_offset,
            image_size,
        );
    }
    return textureLoad(external_texture, vec2<u32>(coords), 0);
}
