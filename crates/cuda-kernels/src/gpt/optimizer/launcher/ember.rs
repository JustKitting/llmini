use cuda_core::DriverError;

use super::super::args::EmberUpdateArgs;
use super::super::ember::{
    EMBER_COLUMN_ROW_TILE, EMBER_THREADS_PER_BLOCK, ember_column_partial_len,
};
use super::OptimizerModule;
use crate::launch::{grid_x_config, linear_config};

impl OptimizerModule {
    pub fn apply_ember_update(&self, args: EmberUpdateArgs<'_>) -> Result<(), DriverError> {
        assert!(args.rows > 0);
        assert!(args.cols > 0);
        assert!(args.beta2_correction > 0.0);
        let len = args.rows as usize * args.cols as usize;
        assert!(args.z_master.len() >= len);
        assert!(args.x_master.len() >= len);
        assert!(args.grad.len() >= len);
        assert!(args.row_second_moment.len() >= args.rows as usize);
        assert!(args.column_second_moment.len() >= args.cols as usize);
        assert!(args.column_partials.len() >= ember_column_partial_len(args.rows, args.cols));
        assert!(!args.normalizer.is_empty());

        self.apply.ember.ember_row_second_moment_kernel(
            args.stream,
            grid_x_config(args.rows, EMBER_THREADS_PER_BLOCK),
            args.grad,
            args.grad_scale,
            &mut *args.row_second_moment,
            args.rows,
            args.cols,
            args.beta2,
        )?;

        let row_tiles = args.rows.div_ceil(EMBER_COLUMN_ROW_TILE);
        let column_tiles = args.cols.div_ceil(EMBER_THREADS_PER_BLOCK);
        self.apply.ember.ember_column_partial_kernel(
            args.stream,
            grid_x_config(row_tiles * column_tiles, EMBER_THREADS_PER_BLOCK),
            args.grad,
            args.grad_scale,
            &mut *args.column_partials,
            args.rows,
            args.cols,
        )?;

        self.apply.ember.ember_column_second_moment_kernel(
            args.stream,
            grid_x_config(args.cols, EMBER_THREADS_PER_BLOCK),
            &*args.column_partials,
            &mut *args.column_second_moment,
            args.rows,
            args.cols,
            args.beta2,
        )?;

        self.apply.ember.ember_normalizer_kernel(
            args.stream,
            grid_x_config(1, EMBER_THREADS_PER_BLOCK),
            &*args.row_second_moment,
            &*args.column_second_moment,
            &mut *args.normalizer,
            args.rows,
            args.cols,
            args.beta2_correction,
        )?;

        self.apply.ember.ember_update_kernel(
            args.stream,
            linear_config(len as u32, EMBER_THREADS_PER_BLOCK),
            &mut *args.z_master,
            &mut *args.x_master,
            args.grad,
            &*args.row_second_moment,
            &*args.column_second_moment,
            &*args.normalizer,
            args.rows,
            args.cols,
            args.grad_scale,
            args.learning_rate,
            args.weight_decay,
            args.beta2_correction,
            args.eps,
            args.average_coefficient,
        )
    }
}
