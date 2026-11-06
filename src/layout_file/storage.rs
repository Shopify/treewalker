//! Array ownership is outside the walk. The pointer/length prefix gives owned
//! and mapped arrays the same branch-free slice accesses; only drop/build differ.

use super::codec::{Codec, Decoder, Encoder, Pod, invalid};
use std::io::{self, Read, Write};
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Arc;

#[repr(C)]
pub(crate) struct Storage<T: Pod> {
    capacity: usize,
    ptr: NonNull<T>,
    len: usize,
    mapping: Option<Arc<Mapping>>,
}

impl<T: Pod> From<Vec<T>> for Storage<T> {
    fn from(values: Vec<T>) -> Self {
        let mut values = ManuallyDrop::new(values);
        Self {
            capacity: values.capacity(),
            ptr: NonNull::new(values.as_mut_ptr()).expect("Vec pointer"),
            len: values.len(),
            mapping: None,
        }
    }
}

impl<T: Pod> Storage<T> {
    #[cfg(all(test, not(miri)))]
    pub(crate) const fn is_mapped(&self) -> bool {
        self.mapping.is_some()
    }

    pub(crate) const fn len(&self) -> usize {
        self.len
    }
    #[inline]
    pub(crate) const fn as_slice(&self) -> &[T] {
        // SAFETY: either a live Vec allocation or aligned, checked Pod bytes in
        // the retained mapping. Mapping callers guarantee immutable file lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn resize(&mut self, len: usize, value: T) {
        self.with_owned(|values| values.resize(len, value));
    }

    pub(crate) fn try_reserve_exact(
        &mut self,
        additional: usize,
    ) -> Result<(), std::collections::TryReserveError> {
        self.with_owned(|values| values.try_reserve_exact(additional))
    }

    fn with_owned<R>(&mut self, f: impl FnOnce(&mut Vec<T>) -> R) -> R {
        assert!(self.mapping.is_none(), "mapped arrays are immutable");
        // Leave an empty owner on unwind before temporarily restoring the Vec.
        let old = ManuallyDrop::new(std::mem::replace(self, Vec::new().into()));
        // SAFETY: this unmapped owner came from a Vec and has not been dropped.
        let mut values = unsafe { Vec::from_raw_parts(old.ptr.as_ptr(), old.len, old.capacity) };
        let result = f(&mut values);
        *self = values.into();
        result
    }
}

impl<T: Pod> Deref for Storage<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: Pod> DerefMut for Storage<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        assert!(self.mapping.is_none(), "mapped arrays are immutable");
        // SAFETY: only an exclusively borrowed owned Vec can reach here.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
impl<T: Pod> Clone for Storage<T> {
    fn clone(&self) -> Self {
        self.mapping.as_ref().map_or_else(
            || self.as_slice().to_vec().into(),
            |map| Self {
                capacity: 0,
                ptr: self.ptr,
                len: self.len,
                mapping: Some(Arc::clone(map)),
            },
        )
    }
}

impl<T: Pod + std::fmt::Debug> std::fmt::Debug for Storage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}
impl<T: Pod> Drop for Storage<T> {
    fn drop(&mut self) {
        if self.mapping.is_none() {
            // SAFETY: owned allocations are transferred from Vec exactly once.
            // Mapped pointers never enter Vec's allocator/deallocation contract.
            drop(unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.capacity) });
        }
    }
}
// SAFETY: owned data obeys Vec's transfer rules; mapped data is immutable for the
// entire lifetime of every Arc owner, as required by map_layout.
unsafe impl<T: Pod + Send> Send for Storage<T> {}
// SAFETY: sharing exposes only slices of Sync elements, never mapped mutation.
unsafe impl<T: Pod + Sync> Sync for Storage<T> {}

impl<T: Pod + Codec> Codec for Storage<T> {
    fn put<W: Write>(&self, w: &mut Encoder<W>) -> io::Result<()> {
        self.len.put(w)?;
        w.align()?;
        T::put_slice(self, w)
    }
    fn get<R: Read>(r: &mut Decoder<R>) -> io::Result<Self> {
        let len = r.length::<T>()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| invalid("layout array size overflow"))?;
        if bytes > r.total.saturating_sub(r.at) {
            return Err(invalid("layout array extent"));
        }
        if let Some(map) = &r.mapping {
            let data = map.bytes(r.at, bytes)?;
            let ptr = data.as_ptr().cast::<T>();
            if !ptr.is_aligned() {
                return Err(invalid("unaligned layout array"));
            }
            r.checksum.write(data);
            r.at += bytes;
            // SAFETY: Mapping's pointer is nonnull, including empty slices.
            let ptr = unsafe { NonNull::new_unchecked(ptr.cast_mut()) };
            return Ok(Self {
                capacity: 0,
                ptr,
                len,
                mapping: Some(Arc::clone(map)),
            });
        }
        Ok(T::get_vec(len, r)?.into())
    }
}

pub(crate) struct Mapping {
    ptr: NonNull<u8>,
    pub(crate) len: usize,
}

// SAFETY: only immutable byte slices are exposed; the unsafe loader requires the
// backing file not be changed or truncated until the last owner is dropped.
unsafe impl Send for Mapping {}
// SAFETY: the same immutable-file lifetime contract permits sharing.
unsafe impl Sync for Mapping {}

unsafe extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        len: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        off: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
}

impl Mapping {
    /// `len` must be the length checked by `open_layout`; the caller guarantees
    /// file immutability until this mapping is dropped.
    pub(crate) unsafe fn new(file: &std::fs::File, len: usize) -> io::Result<Self> {
        use std::os::fd::AsRawFd;
        // SAFETY: read-only private mapping of a nonempty file at offset zero.
        // The caller excludes writes/truncation for its entire lifetime.
        let raw = unsafe { mmap(std::ptr::null_mut(), len, 1, 2, file.as_raw_fd(), 0) };
        if raw as isize == -1 {
            return Err(io::Error::last_os_error());
        }
        let Some(ptr) = NonNull::new(raw.cast()) else {
            // SAFETY: a successful mapping at zero is valid to unmap but cannot
            // back Rust's nonnull slices. This is not expected on supported OSes.
            unsafe { munmap(raw, len) };
            return Err(invalid("null layout mapping"));
        };
        Ok(Self { ptr, len })
    }

    pub(crate) fn bytes(&self, offset: usize, len: usize) -> io::Result<&[u8]> {
        if offset > self.len || len > self.len - offset {
            return Err(invalid("layout mapping extent"));
        }
        // SAFETY: the checked range lies in this live, immutable mapping.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(offset), len) })
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: last owner of the exact pointer and extent returned by mmap.
        unsafe { munmap(self.ptr.as_ptr().cast(), self.len) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_storage_moves_clones_and_grows() {
        let mut values: Storage<u32> = vec![1, 2].into();
        let copy = values.clone();
        values.resize(99, 3);
        values[0] = 0;
        assert_eq!(copy.as_slice(), &[1, 2]);
        assert_eq!(values.len(), 99);
        let empty: Storage<u32> = Vec::new().into();
        assert!(empty.is_empty());
    }
}
