use rust_kernels_cuda::optimizer::MuonSlotDescriptor;

#[derive(Clone, Copy)]
pub(super) struct HostPtrs {
    pub(super) grad: u64,
    pub(super) momentum: u64,
    pub(super) z_master: u64,
    pub(super) x_master: u64,
    pub(super) schedule_amax: u64,
    pub(super) bytes: u64,
    pub(super) scales: u64,
    pub(super) global_scale: u64,
    pub(super) rows: u32,
    pub(super) cols: u32,
    pub(super) learning_rate_multiplier: f32,
    pub(super) qk_clip_factor_offset: u32,
    pub(super) qk_clip_head_dim: u32,
}

impl HostPtrs {
    pub(super) fn descriptor(self) -> MuonSlotDescriptor {
        MuonSlotDescriptor {
            grad: self.grad,
            momentum: self.momentum,
            z_master: self.z_master,
            x_master: self.x_master,
            schedule_amax: self.schedule_amax,
            bytes: self.bytes,
            scales: self.scales,
            global_scale: self.global_scale,
            rows: self.rows,
            cols: self.cols,
            learning_rate_multiplier: self.learning_rate_multiplier,
            qk_clip_factor_offset: self.qk_clip_factor_offset,
            qk_clip_head_dim: self.qk_clip_head_dim,
        }
    }

    pub(super) fn shape(mut self, rows: usize, cols: usize) -> Self {
        self.rows = rows as u32;
        self.cols = cols as u32;
        self
    }

    pub(super) fn learning_rate_multiplier(mut self, value: f32) -> Self {
        self.learning_rate_multiplier = value;
        self
    }

    pub(super) fn qk_clip(mut self, factor_offset: usize, head_dim: usize) -> Self {
        self.qk_clip_factor_offset = factor_offset as u32;
        self.qk_clip_head_dim = head_dim as u32;
        self
    }

    pub(super) fn estimated_polar_work(self) -> u128 {
        let short = self.rows.min(self.cols) as u128;
        let long = self.rows.max(self.cols) as u128;
        short * short * long
    }
}
