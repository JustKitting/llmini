#[derive(Clone, Copy)]
pub struct Coefficients {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

#[inline(always)]
pub fn coefficients(iter: u32) -> Coefficients {
    match iter {
        0 => Coefficients {
            a: 8.287_212 / 1.01,
            b: -23.595_886 / 1.030_301,
            c: 17.300_388 / 1.051_010_1,
        },
        1 => Coefficients {
            a: 4.107_059 / 1.01,
            b: -2.947_85 / 1.030_301,
            c: 0.544_843_1 / 1.051_010_1,
        },
        2 => Coefficients {
            a: 3.948_691 / 1.01,
            b: -2.908_902_2 / 1.030_301,
            c: 0.551_819_15 / 1.051_010_1,
        },
        3 => Coefficients {
            a: 3.318_419_7 / 1.01,
            b: -2.488_488 / 1.030_301,
            c: 0.510_048_9 / 1.051_010_1,
        },
        4 => Coefficients {
            a: 2.300_652 / 1.01,
            b: -1.668_904 / 1.030_301,
            c: 0.418_807_3 / 1.051_010_1,
        },
        _ => Coefficients {
            a: 1.875,
            b: -1.25,
            c: 0.375,
        },
    }
}
