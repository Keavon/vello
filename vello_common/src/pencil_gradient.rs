// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradients whose contours are a pencil of circles.
//!
//! Two generalized circles (circles, lines, or points) with their centers on the gradient line
//! span a pencil: the family of every weighted sum of their implicit equations. Every point of
//! the plane lies on one member, and a [`PencilGradient`] colors it by where that member crosses
//! the gradient line. With both ends lines this is a linear gradient, and with one end a point at
//! the center of the other, a radial one, but the ends can be any such pair, and the family
//! between them varies continuously as either end bends from a line through circles to a point.
//!
//! Everything is in gradient space, where the gradient line is the x axis. The caller works out
//! which crossing of each member's circle belongs to it, since one circle crosses the line twice,
//! and how crossings map to the ramp; [`PencilGeometry`] holds the results.

use crate::kurbo::Point;
use crate::peniko::Gradient;

/// Stands in for an infinite gradient position on contours that cross the gradient line at infinity.
const UNBOUNDED_T: f32 = 1e6;

/// A gradient whose contours are the members of the pencil of circles spanned by two generalized
/// circles centered on the gradient line.
#[derive(Debug, Clone, PartialEq)]
pub struct PencilGradient {
    /// The color stops, extend mode, and interpolation of the gradient. Its kind is unused, since
    /// the geometry places the contours.
    pub ramp: Gradient,
    /// Where the contours lie, in gradient space.
    pub geometry: PencilGeometry,
}

/// The contours of a [`PencilGradient`] in gradient space, where the gradient line is the x axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PencilGeometry {
    /// The start contour as the coefficients `[a, b, c]` of `a(x² + y²) + bx + c = 0`.
    pub start: [f32; 3],
    /// The end contour, in the same form as the start.
    pub end: [f32; 3],
    /// Which of the two places a member's circle crosses the gradient line is the member's own.
    pub selection: PencilSelection,
    /// Crossings that belong past infinity, for pencils whose relevant arc of the gradient line
    /// passes through it.
    pub wrap: Option<PencilWrap>,
    /// The gradient position of a crossing at `x` is `x * t_scale + t_offset`.
    pub t_scale: f32,
    /// See [`Self::t_scale`].
    pub t_offset: f32,
}

/// How a member's crossing is told apart from the other place its circle crosses the gradient line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PencilSelection {
    /// The end contours cross, so every member passes through the same two points, which lie on a
    /// line perpendicular to the gradient line at `radical_x`. Each side of a member's circle is
    /// its own contour, crossing the gradient line on the same side of `radical_x` as the point.
    SameSideAs {
        /// Where the line through the two shared points crosses the gradient line.
        radical_x: f32,
    },
    /// The end contours don't cross, so the members nest around two limit points that split the
    /// gradient line into two arcs. A member's crossing is the one on the arc where
    /// `q[0] n² + q[1] n d + q[2] d²` is least, with the crossing written as `n / d`.
    OnArc {
        /// The coefficients `q` of the quadratic.
        quadratic: [f32; 3],
    },
}

/// For an arc of the gradient line that passes through infinity, where crossings on its far side
/// belong past the end of the gradient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PencilWrap {
    /// A point between the two limit points, which separates the arc's two sides of infinity.
    pub reference_x: f32,
    /// Which side of `reference_x` the gradient's start and end crossings lie on, `1` or `-1`.
    pub band_side: f32,
    /// The gradient position given to crossings on the other side.
    pub t: f32,
}

impl PencilGeometry {
    /// The gradient position at `point` in gradient space, by the same steps as the GPU shader.
    pub fn t(&self, point: Point) -> f32 {
        let (x, y) = (point.x as f32, point.y as f32);
        let implicit = |[a, b, c]: [f32; 3]| a * (x * x + y * y) + b * x + c;

        // Normalized so the products below stay in range far from the origin
        let (start_value, end_value) = (implicit(self.start), implicit(self.end));
        let scale = start_value.abs().max(end_value.abs());
        if scale == 0.0 {
            return 0.0;
        }
        let (start_value, end_value) = (start_value / scale, end_value / scale);

        // The member through the point keeps the ratio of the two contours' values, so it
        // crosses the gradient line at the roots of this quadratic
        let [start_a, start_b, start_c] = self.start;
        let [end_a, end_b, end_c] = self.end;
        let a = end_value * start_a - start_value * end_a;
        let b = end_value * start_b - start_value * end_b;
        let c = end_value * start_c - start_value * end_c;
        let roots = homogeneous_roots(a, b, c);

        let crossing = match self.selection {
            PencilSelection::SameSideAs { radical_x } => {
                let side = |(n, d): (f32, f32)| (n - radical_x * d) * d * (x - radical_x);
                let (first, second) = (side(roots[0]), side(roots[1]));
                // A point on the dividing line is on the member that is a line, the finite root
                let first_wins = if first == second {
                    roots[0].1.abs() >= roots[1].1.abs()
                } else {
                    first > second
                };
                if first_wins { roots[0] } else { roots[1] }
            }
            PencilSelection::OnArc {
                quadratic: [q2, q1, q0],
            } => {
                let arc = |(n, d): (f32, f32)| q2 * n * n + q1 * n * d + q0 * d * d;
                if arc(roots[0]) <= arc(roots[1]) {
                    roots[0]
                } else {
                    roots[1]
                }
            }
        };

        let (n, d) = crossing;
        if let Some(wrap) = self.wrap
            && (n - wrap.reference_x * d) * d * wrap.band_side < 0.0
        {
            return wrap.t;
        }

        if d == 0.0 {
            // Like WGSL's `sign`, which is zero for zero
            let direction = n * self.t_scale;
            return if direction > 0.0 {
                UNBOUNDED_T
            } else if direction < 0.0 {
                -UNBOUNDED_T
            } else {
                0.0
            };
        }
        n / d * self.t_scale + self.t_offset
    }
}

/// The roots of `a x² + b x + c = 0` as `(numerator, denominator)` pairs, each scaled so its
/// larger part has a magnitude of 1, with a vanishing `a` giving a root at infinity.
fn homogeneous_roots(a: f32, b: f32, c: f32) -> [(f32, f32); 2] {
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    let root = discriminant.sqrt();
    let q = -0.5 * (b + if b >= 0.0 { root } else { -root });

    let normalize = |(n, d): (f32, f32)| {
        let scale = n.abs().max(d.abs());
        if scale == 0.0 {
            (n, d)
        } else {
            (n / scale, d / scale)
        }
    };
    [normalize((q, a)), normalize((c, q))]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The radial gradient around the origin with radius 1, as a pencil: the start contour is the
    /// point at the origin and the end contour is the unit circle.
    fn radial() -> PencilGeometry {
        PencilGeometry {
            start: [1.0, 0.0, 0.0],
            end: [1.0, 0.0, -1.0],
            selection: PencilSelection::OnArc {
                quadratic: [0.0, -1.0, 0.0],
            },
            wrap: None,
            t_scale: 1.0,
            t_offset: 0.0,
        }
    }

    #[test]
    fn radial_pencil_spaces_contours_by_radius() {
        let geometry = radial();
        for radius in [0.25_f64, 0.5, 1.0, 2.0] {
            for angle in [0.0_f64, 1.0, 2.0, 4.0] {
                let point = Point::new(radius * angle.cos(), radius * angle.sin());
                assert!((geometry.t(point) - radius as f32).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn linear_pencil_spaces_contours_along_the_line() {
        let geometry = PencilGeometry {
            start: [0.0, -2.0, 0.0],
            end: [0.0, 2.0, -2.0],
            selection: PencilSelection::OnArc {
                quadratic: [0.0, 0.0, -1.0],
            },
            wrap: None,
            t_scale: 1.0,
            t_offset: 0.0,
        };
        for point in [
            Point::new(0.25, 0.0),
            Point::new(0.75, 3.0),
            Point::new(-1.0, -2.0),
        ] {
            assert!((geometry.t(point) - point.x as f32).abs() < 1e-5);
        }
    }
}
