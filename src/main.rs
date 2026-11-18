mod checkpoint;
mod env_file;
mod fs_utils;
mod reference_validation;
mod time_utils;
mod training;
mod upload;

use std::error::Error;

type AppResult<T = ()> = Result<T, Box<dyn Error>>;

fn main() -> AppResult {
    training::launch_from_args()
}
