use super::super::output::CudaValidOutput;

metric_fields! {
    ValidMetricField, VALID_METRIC_FIELDS, valid_metric_specs, ValidMetricSpec,
    CudaValidOutput, ValidMetricField::value, "" {
        Loss => ("Validation loss", None, false),
        BitsPerByte => ("Validation bits per byte", None, false),
        EvalElapsed => ("Eval elapsed", Some("s"), false),
        WindowCount => ("Val windows", None, true),
        TargetTokenCount => ("Val target tokens", None, true),
        TargetByteCount => ("Val target bytes", None, true),
        CompletedSteps => ("Completed steps", None, true),
    }
}

impl ValidMetricField {
    fn value(self, item: &CudaValidOutput) -> f64 {
        match self {
            Self::Loss => item.val_loss as f64,
            Self::BitsPerByte => item.val_bits_per_byte,
            Self::EvalElapsed => item.eval_elapsed_s,
            Self::WindowCount => item.window_count as f64,
            Self::TargetTokenCount => item.target_token_count as f64,
            Self::TargetByteCount => item.target_byte_count as f64,
            Self::CompletedSteps => item.completed_steps as f64,
        }
    }
}
