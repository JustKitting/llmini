use std::error::Error;

#[path = "muon/buffers.rs"]
mod buffers;
#[path = "muon/fixture.rs"]
mod fixture;
#[path = "muon/nonconstant.rs"]
mod nonconstant;
#[path = "muon/tma_finish.rs"]
mod tma_finish;
#[path = "muon/tma_prepare.rs"]
mod tma_prepare;

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn muon_mega_update_matches_first_iteration_recurrence() -> Result<(), Box<dyn Error>> {
    fixture::run_first_iteration_case(64, 64)
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn muon_mega_update_matches_tall_rectangular_recurrence() -> Result<(), Box<dyn Error>> {
    fixture::run_first_iteration_case(64, 32)
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn muon_mega_update_matches_wide_rectangular_recurrence() -> Result<(), Box<dyn Error>> {
    fixture::run_first_iteration_case(32, 64)
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn muon_mega_update_matches_nonconstant_wide_recurrence() -> Result<(), Box<dyn Error>> {
    nonconstant::run_wide_case()
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn muon_tma_finish_retains_next_schedule_amax() -> Result<(), Box<dyn Error>> {
    tma_finish::run_schedule_amax_case()
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn muon_tma_split_finish_matches_cooperative_reference() -> Result<(), Box<dyn Error>> {
    tma_finish::run_split_matches_reference_case()
}

#[ignore = "requires generated sm_120a PTX"]
#[test]
fn muon_tma_split_prepare_matches_cooperative_reference() -> Result<(), Box<dyn Error>> {
    tma_prepare::run_split_matches_reference_case()
}
