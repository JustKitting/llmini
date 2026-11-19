use rust_kernels_cuda::linear_backward::{
    LinearBackwardMsEdenScratch, LinearBackwardTmaScratch, MsEdenOperandScratch,
    MsEdenTransposeScratch,
};

pub(crate) fn reborrow_ms_eden<'a, 'b>(
    scratch: &'b mut LinearBackwardMsEdenScratch<'a>,
) -> LinearBackwardMsEdenScratch<'b> {
    LinearBackwardMsEdenScratch {
        e_h: reborrow_operand(&mut scratch.e_h),
        weight_t_h: reborrow_operand(&mut scratch.weight_t_h),
        e_t_h: MsEdenTransposeScratch {
            bytes: &mut *scratch.e_t_h.bytes,
            scales: &mut *scratch.e_t_h.scales,
        },
        input_t_h: reborrow_operand(&mut scratch.input_t_h),
        tma: reborrow_tma(&mut scratch.tma),
    }
}

fn reborrow_operand<'a, 'b>(operand: &'b mut MsEdenOperandScratch<'a>) -> MsEdenOperandScratch<'b> {
    MsEdenOperandScratch {
        bytes: &mut *operand.bytes,
        scales: &mut *operand.scales,
        chunk_amax: &mut *operand.chunk_amax,
        global_scale: &mut *operand.global_scale,
    }
}

fn reborrow_tma<'a, 'b>(tma: &'b mut LinearBackwardTmaScratch<'a>) -> LinearBackwardTmaScratch<'b> {
    LinearBackwardTmaScratch {
        descriptors: &mut *tma.descriptors,
    }
}
