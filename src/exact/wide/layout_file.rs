use super::{Wide, WideLeaf};
use crate::layout_file::codec::fields;

fields!(Wide { limbs, low });
fields!(WideLeaf { first, digits }, 16);
