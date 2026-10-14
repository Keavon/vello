// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::fine::common::gradient::SimdGradientKind;
use crate::kurbo::Point;
use vello_common::fearless_simd::{Simd, SimdBase, f32x8};
use vello_common::pencil_gradient::PencilGeometry;

#[derive(Debug)]
pub(crate) struct SimdPencilKind<S: Simd> {
    geometry: PencilGeometry,
    simd: S,
}

impl<S: Simd> SimdPencilKind<S> {
    pub(crate) fn new(simd: S, geometry: &PencilGeometry) -> Self {
        Self {
            geometry: *geometry,
            simd,
        }
    }
}

impl<S: Simd> SimdGradientKind<S> for SimdPencilKind<S> {
    // Each lane takes the scalar path, since the root selection branches per pixel.
    #[inline(always)]
    fn cur_pos(&self, x_pos: f32x8<S>, y_pos: f32x8<S>) -> f32x8<S> {
        let (x, y) = (x_pos.as_slice(), y_pos.as_slice());
        let t: [f32; 8] = core::array::from_fn(|lane| {
            self.geometry
                .t(Point::new(f64::from(x[lane]), f64::from(y[lane])))
        });
        f32x8::from_slice(self.simd, &t)
    }
}
