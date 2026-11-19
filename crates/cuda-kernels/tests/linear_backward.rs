use std::error::Error;

mod common;
#[path = "linear_backward/ms_eden.rs"]
mod ms_eden;
#[path = "linear_backward/quartet.rs"]
mod quartet;

type TestResult = Result<(), Box<dyn Error>>;

const TOKEN_COUNT: usize = 128;
const INPUT_DIM: usize = 128;
const OUTPUT_DIM: usize = 128;
const TOLERANCE: f32 = 1.0e-7;
