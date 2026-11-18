use std::{env, fs, path::Path};

const DEFAULT_M_REPEAT: u32 = 2;
const DEFAULT_N_REPEAT: u32 = 8;
const DEFAULT_TILE_M: u32 = 128;
const DEFAULT_TILE_N: u32 = 128;
const DEFAULT_TILE_K: u32 = 128;
const DEFAULT_TMA_STAGES: u32 = 5;
const DEFAULT_N_LAYOUT: Nvfp4NLayout = Nvfp4NLayout::PairInterleaved;
const DEFAULT_K_ATOM_ORDER: Nvfp4KAtomOrder = Nvfp4KAtomOrder::Forward;
const DEFAULT_M_AXIS_ORDER: Nvfp4AxisOrder = Nvfp4AxisOrder::Normal;
const DEFAULT_N_AXIS_ORDER: Nvfp4AxisOrder = Nvfp4AxisOrder::Normal;
const MMA_M: u32 = 16;
const MMA_N: u32 = 8;
const WARP_THREADS: u32 = 32;
const HARDWARE_REGISTER_LIMIT: u32 = 255;
const ACCUMULATOR_REGS: u32 = 4;
const MAX_GENERATED_ACCUMULATORS: u32 = 4096;
const LAUNCH_BOUNDS_THREAD_COUNTS: [u32; 9] = [64, 96, 128, 160, 192, 224, 256, 288, 384];

#[derive(Clone, Copy)]
enum Nvfp4NLayout {
    PairInterleaved,
    WarpContiguous,
    RepeatInterleaved,
    PairReverseGroups,
    PairOuterGroups,
}

impl Nvfp4NLayout {
    const fn as_u32(self) -> u32 {
        match self {
            Self::PairInterleaved => 0,
            Self::WarpContiguous => 1,
            Self::RepeatInterleaved => 2,
            Self::PairReverseGroups => 3,
            Self::PairOuterGroups => 4,
        }
    }
}

#[derive(Clone, Copy)]
enum Nvfp4KAtomOrder {
    Forward,
    Reverse,
}

impl Nvfp4KAtomOrder {
    const fn as_u32(self) -> u32 {
        match self {
            Self::Forward => 0,
            Self::Reverse => 1,
        }
    }
}

#[derive(Clone, Copy)]
enum Nvfp4AxisOrder {
    Normal,
    Reverse,
    PairReverseGroups,
    PairOuterGroups,
    WithinPairReverse,
    EvenOdd,
    OddEven,
    Outer,
}

impl Nvfp4AxisOrder {
    const fn at(self, extent: u32, position: u32) -> u32 {
        axis_order_at(extent, position, self)
    }
}

pub fn emit_nvfp4_config(out_dir: &Path) {
    println!("cargo:rerun-if-env-changed=NVFP4_M_REPEAT");
    println!("cargo:rerun-if-env-changed=NVFP4_N_REPEAT");
    println!("cargo:rerun-if-env-changed=NVFP4_TILE_M");
    println!("cargo:rerun-if-env-changed=NVFP4_TILE_N");
    println!("cargo:rerun-if-env-changed=NVFP4_TILE_K");
    println!("cargo:rerun-if-env-changed=NVFP4_TMA_STAGES");
    println!("cargo:rerun-if-env-changed=NVFP4_N_LAYOUT");
    println!("cargo:rerun-if-env-changed=NVFP4_K_ATOM_ORDER");
    println!("cargo:rerun-if-env-changed=NVFP4_M_AXIS_ORDER");
    println!("cargo:rerun-if-env-changed=NVFP4_N_AXIS_ORDER");
    for threads in LAUNCH_BOUNDS_THREAD_COUNTS {
        println!("cargo:rustc-check-cfg=cfg(nvfp4_launch_bounds_{threads})");
    }

    let m_repeat = env_u32("NVFP4_M_REPEAT", DEFAULT_M_REPEAT);
    let n_repeat = env_u32("NVFP4_N_REPEAT", DEFAULT_N_REPEAT);
    let tile_m = env_u32("NVFP4_TILE_M", DEFAULT_TILE_M);
    let tile_n = env_u32("NVFP4_TILE_N", DEFAULT_TILE_N);
    let tile_k = env_u32("NVFP4_TILE_K", DEFAULT_TILE_K);
    let tma_stages = env_u32("NVFP4_TMA_STAGES", DEFAULT_TMA_STAGES);
    let n_layout = env_n_layout("NVFP4_N_LAYOUT", DEFAULT_N_LAYOUT).as_u32();
    let k_atom_order = env_k_atom_order("NVFP4_K_ATOM_ORDER", DEFAULT_K_ATOM_ORDER).as_u32();
    let m_axis_order = env_axis_order("NVFP4_M_AXIS_ORDER", DEFAULT_M_AXIS_ORDER);
    let n_axis_order = env_axis_order("NVFP4_N_AXIS_ORDER", DEFAULT_N_AXIS_ORDER);
    let launch_threads = emit_geometry_config(
        out_dir,
        "nvfp4_config.rs",
        [m_repeat, n_repeat],
        [tile_m, tile_n, tile_k],
        tma_stages,
        (n_layout, k_atom_order, m_axis_order, n_axis_order),
    );
    println!("cargo:rustc-cfg=nvfp4_launch_bounds_{launch_threads}");
    // Polar updates have a compact source epilogue and benefit from two CTAs.
    // Projection/Gram families keep their independent accepted geometry.
    let polar_threads = emit_geometry_config(
        out_dir,
        "nvfp4_polar_config.rs",
        [2, 4],
        [128, 64, 128],
        3,
        (
            Nvfp4NLayout::PairInterleaved.as_u32(),
            Nvfp4KAtomOrder::Forward.as_u32(),
            Nvfp4AxisOrder::Normal,
            Nvfp4AxisOrder::Normal,
        ),
    );
    assert_eq!(polar_threads, 288);
}

fn emit_geometry_config(
    out_dir: &Path,
    filename: &str,
    [m_repeat, n_repeat]: [u32; 2],
    [tile_m, tile_n, tile_k]: [u32; 3],
    tma_stages: u32,
    (n_layout, k_atom_order, m_axis_order, n_axis_order): (
        u32,
        u32,
        Nvfp4AxisOrder,
        Nvfp4AxisOrder,
    ),
) -> u32 {
    assert!(
        tile_k % 64 == 0,
        "NVFP4_TILE_K must be divisible by the SM120 NVFP4 MMA K atom (64), got {tile_k}"
    );
    let k_atoms = tile_k / 64;
    let max_accumulators = HARDWARE_REGISTER_LIMIT / ACCUMULATOR_REGS;
    let requested_accumulators = m_repeat
        .checked_mul(n_repeat)
        .expect("NVFP4_M_REPEAT * NVFP4_N_REPEAT overflows u32");
    let accumulator_pool_len = requested_accumulators
        .max(max_accumulators)
        .min(MAX_GENERATED_ACCUMULATORS);

    let m_axis = repeat_markers(m_repeat);
    let n_axis = repeat_markers(n_repeat);
    let m_issue_axis = repeat_axis(m_repeat, m_axis_order);
    let n_issue_axis = repeat_axis(n_repeat, n_axis_order);
    let k_issue_axis = k_atom_axis(k_atoms, k_atom_order);
    let launch_threads = launch_bounds_threads(tile_m, tile_n, m_repeat, n_repeat);
    assert!(
        LAUNCH_BOUNDS_THREAD_COUNTS.contains(&launch_threads),
        "generated launch bounds thread count {launch_threads} is not listed in LAUNCH_BOUNDS_THREAD_COUNTS"
    );
    let accumulators = accumulator_names(accumulator_pool_len);
    let mut generated = String::new();
    generated.push_str("// Generated by build.rs. Configure with NVFP4_* environment variables.\n");
    generated.push_str(&format!(
        "pub const NVFP4_REQUESTED_TILE_M: u32 = {tile_m};\n"
    ));
    generated.push_str(&format!(
        "pub const NVFP4_REQUESTED_TILE_N: u32 = {tile_n};\n"
    ));
    generated.push_str(&format!("pub const NVFP4_TILE_K: u32 = {tile_k};\n"));
    generated.push_str(&format!(
        "pub const NVFP4_TMA_STAGES: u32 = {tma_stages};\n"
    ));
    generated.push_str(&format!("pub const NVFP4_N_LAYOUT: u32 = {n_layout};\n"));
    generated.push_str(&format!(
        "pub const NVFP4_K_ATOM_ORDER: u32 = {k_atom_order};\n"
    ));
    generated.push_str(&format!(
        "pub const NVFP4_MAX_ACCUMULATORS_PER_THREAD: u32 = {max_accumulators};\n"
    ));
    generated.push_str("macro_rules! nvfp4_warp_repeat_shape {\n");
    generated.push_str("    ($use_shape:ident $(, $arg:tt)*) => {\n");
    generated.push_str("        $use_shape!([");
    generated.push_str(&m_axis);
    generated.push_str("], [");
    generated.push_str(&n_axis);
    generated.push_str("] $(, $arg)*);\n");
    generated.push_str("    };\n");
    generated.push_str("}\n\n");
    generated.push_str("macro_rules! nvfp4_warp_issue_axes {\n");
    generated.push_str("    ($use_axes:ident $(, $arg:tt)*) => {\n");
    generated.push_str("        $use_axes!([");
    generated.push_str(&m_issue_axis);
    generated.push_str("], [");
    generated.push_str(&n_issue_axis);
    generated.push_str("] $(, $arg)*);\n");
    generated.push_str("    };\n");
    generated.push_str("}\n\n");
    generated.push_str("macro_rules! nvfp4_k_atom_axis {\n");
    generated.push_str("    ($use_axis:ident $(, $arg:tt)*) => {\n");
    generated.push_str("        $use_axis!([");
    generated.push_str(&k_issue_axis);
    generated.push_str("] $(, $arg)*);\n");
    generated.push_str("    };\n");
    generated.push_str("}\n\n");
    generated.push_str("macro_rules! nvfp4_accumulator_pool {\n");
    generated.push_str("    ($use_pool:ident $(, $arg:tt)*) => {\n");
    generated.push_str("        $use_pool!([");
    generated.push_str(&accumulators);
    generated.push_str("] $(, $arg)*);\n");
    generated.push_str("    };\n");
    generated.push_str("}\n");

    fs::write(out_dir.join(filename), generated).expect("write NVFP4 geometry config");
    launch_threads
}

fn env_u32(name: &str, default: u32) -> u32 {
    match env::var(name) {
        Ok(value) => {
            let parsed: u32 = value.parse().unwrap_or_else(|_| {
                panic!("{name} must be an unsigned integer, got {value:?}");
            });
            assert!(parsed > 0, "{name} must be greater than zero");
            parsed
        }
        Err(_) => default,
    }
}

fn env_n_layout(name: &str, default: Nvfp4NLayout) -> Nvfp4NLayout {
    let Ok(value) = env::var(name) else {
        return default;
    };
    parse_n_layout(&value).unwrap_or_else(|| {
        panic!(
            "{name} must be pair_interleaved, warp_contiguous, repeat_interleaved, pair_reverse_groups, or pair_outer_groups, got {value:?}"
        )
    })
}

fn parse_n_layout(value: &str) -> Option<Nvfp4NLayout> {
    match value {
        "0" | "pair_interleaved" | "colfax" => Some(Nvfp4NLayout::PairInterleaved),
        "1" | "warp_contiguous" => Some(Nvfp4NLayout::WarpContiguous),
        "2" | "repeat_interleaved" => Some(Nvfp4NLayout::RepeatInterleaved),
        "3" | "pair_reverse_groups" => Some(Nvfp4NLayout::PairReverseGroups),
        "4" | "pair_outer_groups" => Some(Nvfp4NLayout::PairOuterGroups),
        _ => None,
    }
}

fn env_k_atom_order(name: &str, default: Nvfp4KAtomOrder) -> Nvfp4KAtomOrder {
    let Ok(value) = env::var(name) else {
        return default;
    };
    parse_k_atom_order(&value)
        .unwrap_or_else(|| panic!("{name} must be forward or reverse, got {value:?}"))
}

fn parse_k_atom_order(value: &str) -> Option<Nvfp4KAtomOrder> {
    match value {
        "0" | "forward" => Some(Nvfp4KAtomOrder::Forward),
        "1" | "reverse" => Some(Nvfp4KAtomOrder::Reverse),
        _ => None,
    }
}

fn env_axis_order(name: &str, default: Nvfp4AxisOrder) -> Nvfp4AxisOrder {
    let Ok(value) = env::var(name) else {
        return default;
    };
    parse_axis_order(&value).unwrap_or_else(|| {
        panic!(
            "{name} must be normal, reverse, pair_reverse_groups, pair_outer_groups, within_pair_reverse, even_odd, odd_even, or outer, got {value:?}"
        )
    })
}

fn parse_axis_order(value: &str) -> Option<Nvfp4AxisOrder> {
    match value {
        "normal" => Some(Nvfp4AxisOrder::Normal),
        "reverse" => Some(Nvfp4AxisOrder::Reverse),
        "pair_reverse_groups" => Some(Nvfp4AxisOrder::PairReverseGroups),
        "pair_outer_groups" => Some(Nvfp4AxisOrder::PairOuterGroups),
        "within_pair_reverse" => Some(Nvfp4AxisOrder::WithinPairReverse),
        "even_odd" => Some(Nvfp4AxisOrder::EvenOdd),
        "odd_even" => Some(Nvfp4AxisOrder::OddEven),
        "outer" => Some(Nvfp4AxisOrder::Outer),
        _ => None,
    }
}

fn repeat_markers(count: u32) -> String {
    (0..count).map(|_| "_").collect::<Vec<_>>().join(", ")
}

fn repeat_axis(count: u32, order: Nvfp4AxisOrder) -> String {
    (0..count)
        .map(|position| order.at(count, position).to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn k_atom_axis(count: u32, order: u32) -> String {
    match order {
        0 => (0..count)
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        1 => (0..count)
            .rev()
            .map(|index| index.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        _ => unreachable!("validated K atom order"),
    }
}

fn accumulator_names(count: u32) -> String {
    (0..count)
        .map(|index| format!("acc{index:04}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn launch_bounds_threads(tile_m: u32, tile_n: u32, m_repeat: u32, n_repeat: u32) -> u32 {
    let m_per_warp = m_repeat * MMA_M;
    let n_per_warp = n_repeat * MMA_N;
    let warp_tiles_m = ceil_div_u32(tile_m, m_per_warp);
    let warp_tiles_n = ceil_div_u32(tile_n, n_per_warp);
    warp_tiles_m * warp_tiles_n * WARP_THREADS + WARP_THREADS
}

#[derive(Clone, Copy)]
enum GroupOrder {
    Reverse,
    Outer,
}

const fn axis_order_at(extent: u32, position: u32, order: Nvfp4AxisOrder) -> u32 {
    match order {
        Nvfp4AxisOrder::Normal => position,
        Nvfp4AxisOrder::Reverse => extent - 1 - position,
        Nvfp4AxisOrder::PairReverseGroups => {
            pair_group_axis_at(extent, position, GroupOrder::Reverse)
        }
        Nvfp4AxisOrder::PairOuterGroups => pair_group_axis_at(extent, position, GroupOrder::Outer),
        Nvfp4AxisOrder::WithinPairReverse => within_pair_reverse_at(extent, position),
        Nvfp4AxisOrder::EvenOdd => parity_axis_at(extent, position, 0),
        Nvfp4AxisOrder::OddEven => parity_axis_at(extent, position, 1),
        Nvfp4AxisOrder::Outer => outer_at(extent, position),
    }
}

const fn pair_group_axis_at(extent: u32, position: u32, order: GroupOrder) -> u32 {
    let groups = ceil_div_u32(extent, 2);
    let mut group_position = 0;
    let mut remaining = position;
    while group_position < groups {
        let group = match order {
            GroupOrder::Reverse => groups - 1 - group_position,
            GroupOrder::Outer => outer_at(groups, group_position),
        };
        let first = group * 2;
        if first < extent {
            if remaining == 0 {
                return first;
            }
            remaining -= 1;
        }
        if first + 1 < extent {
            if remaining == 0 {
                return first + 1;
            }
            remaining -= 1;
        }
        group_position += 1;
    }
    panic!("axis position outside extent")
}

const fn within_pair_reverse_at(extent: u32, position: u32) -> u32 {
    let group = position / 2;
    let lane = position - group * 2;
    let first = group * 2;
    if first + 1 < extent {
        if lane == 0 { first + 1 } else { first }
    } else {
        first
    }
}

const fn parity_axis_at(extent: u32, position: u32, first_parity: u32) -> u32 {
    let first_count = if first_parity == 0 {
        ceil_div_u32(extent, 2)
    } else {
        extent / 2
    };
    if position < first_count {
        first_parity + position * 2
    } else {
        (1 - first_parity) + (position - first_count) * 2
    }
}

const fn outer_at(extent: u32, position: u32) -> u32 {
    if position & 1 == 0 {
        position / 2
    } else {
        extent - 1 - position / 2
    }
}

const fn ceil_div_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        0
    } else {
        ((value - 1) / divisor) + 1
    }
}
