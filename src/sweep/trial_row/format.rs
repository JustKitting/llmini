use super::super::fmt;
use super::super::history::Trial;

pub(super) fn header() -> &'static str {
    "status\tval_loss\tcompleted_steps\tbatch_size\tn_layer\tn_embd\tn_head\tmuon_phases\tmuon_blocks\tlr_scale\tadam_lr_scale\tnextlat_lr_scale\twarmup_steps\tstart_ratio\tamuse_beta1\tamuse_rho\tlog_path\telapsed_s\tscreen_val_loss\tscreen_completed_steps\tscreen_elapsed_s\tscreen_reason"
}

pub(super) fn format_trial(trial: &Trial) -> String {
    let c = &trial.candidate;
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}",
        trial.status,
        fmt::optional_f64_6_or_nan(trial.val_loss),
        fmt::optional_usize(trial.completed_steps),
        c.batch_size,
        c.n_layer,
        c.n_embd,
        c.n_head,
        c.muon_phases,
        c.muon_blocks,
        c.lr_scale,
        c.adam_lr_scale,
        c.nextlat_lr_scale,
        c.warmup_steps,
        c.start_ratio,
        c.amuse_beta1,
        c.amuse_rho,
        trial.log_path.display(),
        fmt::optional_f64_6(trial.elapsed_s),
        fmt::optional_f64_6(trial.screen_val_loss),
        fmt::optional_usize(trial.screen_completed_steps),
        fmt::optional_f64_6(trial.screen_elapsed_s),
        trial.screen_reason.as_deref().unwrap_or_default()
    )
}
