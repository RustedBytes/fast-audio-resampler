use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

#[derive(Debug)]
pub(crate) struct AlignedVec<T: Copy + Default, const ALIGN: usize> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    _value: PhantomData<T>,
}

impl<T: Copy + Default, const ALIGN: usize> AlignedVec<T, ALIGN> {
    pub(crate) fn zeroed(len: usize) -> Self {
        assert!(ALIGN.is_power_of_two());
        assert!(ALIGN >= align_of::<T>());
        let size = len
            .checked_mul(size_of::<T>())
            .expect("aligned allocation overflow");
        let layout = Layout::from_size_align(size.max(1), ALIGN).expect("invalid aligned layout");
        let raw = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(raw.cast::<T>()).unwrap_or_else(|| handle_alloc_error(layout));
        Self {
            ptr,
            len,
            layout,
            _value: PhantomData,
        }
    }

    pub(crate) fn from_slice(values: &[T]) -> Self {
        let mut out = Self::zeroed(values.len());
        out.copy_from_slice(values);
        out
    }
}

impl<T: Copy + Default, const ALIGN: usize> Clone for AlignedVec<T, ALIGN> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: Copy + Default, const ALIGN: usize> Deref for AlignedVec<T, ALIGN> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy + Default, const ALIGN: usize> DerefMut for AlignedVec<T, ALIGN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy + Default, const ALIGN: usize> Drop for AlignedVec<T, ALIGN> {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout) };
    }
}
