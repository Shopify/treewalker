//! Today's loader, model and kernel, which the new pipeline replaces piece by piece:
//! they serve [`crate::Predictor`] until the new kernel lands, and the research API
//! until it moves to the new kernel too.

pub(crate) mod forest;
pub(crate) mod mask;
pub(crate) mod parser;
pub(crate) mod predict;
