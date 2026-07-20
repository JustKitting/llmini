use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    synth_prep::fineweb::parse_data()
}
