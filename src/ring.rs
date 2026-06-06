#[derive(Debug, Clone)]
pub(crate) struct RingBuffer<T: Copy + Default> {
    data: Vec<T>,
    start_frame: i64,
    len: usize,
}

impl<T: Copy + Default> RingBuffer<T> {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity.max(1)],
            start_frame: 0,
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn start_frame(&self) -> i64 {
        self.start_frame
    }

    #[inline]
    pub(crate) fn end_frame(&self) -> i64 {
        self.start_frame + self.len as i64
    }

    pub(crate) fn ensure_capacity(&mut self, capacity: usize) {
        if capacity <= self.data.len() {
            return;
        }
        let mut new_data = vec![T::default(); capacity.next_power_of_two()];
        let new_capacity = new_data.len();
        let new_offset = self.start_frame as usize % new_capacity;
        for i in 0..self.len {
            new_data[(new_offset + i) % new_capacity] =
                self.get(self.start_frame + i as i64).unwrap_or_default();
        }
        self.data = new_data;
    }

    pub(crate) fn push(&mut self, value: T) {
        if self.len < self.data.len() {
            let index = self.physical_index(self.len);
            self.data[index] = value;
            self.len += 1;
            return;
        }

        let index = self.physical_index(0);
        self.data[index] = value;
        self.start_frame += 1;
    }

    #[inline(always)]
    pub(crate) fn get(&self, absolute_frame: i64) -> Option<T> {
        if absolute_frame < self.start_frame || absolute_frame >= self.end_frame() {
            return None;
        }
        let local = (absolute_frame - self.start_frame) as usize;
        Some(self.data[self.physical_index(local)])
    }

    pub(crate) fn discard_before(&mut self, absolute_frame: i64) {
        let keep_from = absolute_frame.clamp(self.start_frame, self.end_frame());
        let remove = (keep_from - self.start_frame) as usize;
        self.start_frame = keep_from;
        self.len -= remove;
    }

    #[inline(always)]
    fn physical_index(&self, local: usize) -> usize {
        (self.start_frame as usize + local) % self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_absolute_lookup_across_wrap_and_growth() {
        let mut ring = RingBuffer::<i32>::with_capacity(4);
        for value in 0..6 {
            ring.push(value);
        }
        assert_eq!(ring.start_frame(), 2);
        assert_eq!(ring.get(1), None);
        assert_eq!(ring.get(2), Some(2));
        assert_eq!(ring.get(5), Some(5));

        ring.ensure_capacity(10);
        for value in 6..10 {
            ring.push(value);
        }
        assert_eq!(ring.get(2), Some(2));
        assert_eq!(ring.get(9), Some(9));
    }
}
