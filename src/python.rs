use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{
    Error, FirBackend, ProcessStats as RustProcessStats, Quality, Resampler, ResamplerConfig,
};

const QUALITY_VALUES: [&str; 3] = ["fast", "balanced", "best"];
const BACKEND_VALUES: [&str; 7] = [
    "auto", "scalar", "avx2", "avx2+fma", "avx512", "neon", "rvv",
];

#[pyclass(name = "ProcessStats", module = "fast_audio_resampler", frozen)]
#[derive(Debug)]
struct PyProcessStats {
    input_frames: usize,
    output_frames: usize,
    backend: String,
}

#[pymethods]
impl PyProcessStats {
    #[getter]
    fn input_frames(&self) -> usize {
        self.input_frames
    }

    #[getter]
    fn output_frames(&self) -> usize {
        self.output_frames
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessStats(input_frames={}, output_frames={}, backend={:?})",
            self.input_frames, self.output_frames, self.backend
        )
    }
}

impl From<RustProcessStats> for PyProcessStats {
    fn from(stats: RustProcessStats) -> Self {
        Self {
            input_frames: stats.input_frames,
            output_frames: stats.output_frames,
            backend: stats.backend.name().to_owned(),
        }
    }
}

#[pyclass(name = "F32Resampler", module = "fast_audio_resampler", unsendable)]
#[derive(Debug)]
struct PyF32Resampler {
    inner: Resampler<f32>,
    channels: usize,
}

#[pymethods]
impl PyF32Resampler {
    #[new]
    #[pyo3(signature = (
        input_rate,
        output_rate,
        channels,
        quality = None,
        backend = None,
        max_input_frames_per_chunk = None
    ))]
    fn new(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        quality: Option<&str>,
        backend: Option<&str>,
        max_input_frames_per_chunk: Option<usize>,
    ) -> PyResult<Self> {
        let config = config_from_parts(
            input_rate,
            output_rate,
            channels,
            quality,
            backend,
            max_input_frames_per_chunk,
        )?;

        Ok(Self {
            inner: Resampler::<f32>::new(config).map_err(to_py_err)?,
            channels,
        })
    }

    /// Process interleaved f32 samples and return `(output, stats)`.
    fn process(&mut self, input: Vec<f32>) -> PyResult<(Vec<f32>, PyProcessStats)> {
        let input_frames = input.len() / self.channels;
        let mut output = Vec::with_capacity(self.inner.required_output_capacity(input_frames));
        let stats = self
            .inner
            .process(&input, &mut output)
            .map_err(to_py_err)?
            .into();
        Ok((output, stats))
    }

    /// Finish the stream and return `(output, stats)`.
    fn finish(&mut self) -> PyResult<(Vec<f32>, PyProcessStats)> {
        let mut output = Vec::new();
        let stats = self.inner.finish(&mut output).map_err(to_py_err)?.into();
        Ok((output, stats))
    }

    /// Alias for `finish`.
    fn flush(&mut self) -> PyResult<(Vec<f32>, PyProcessStats)> {
        self.finish()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn required_output_capacity(&self, input_frames: usize) -> usize {
        self.inner.required_output_capacity(input_frames)
    }

    #[getter]
    fn selected_backend(&self) -> &'static str {
        self.inner.selected_backend().name()
    }

    fn __repr__(&self) -> String {
        format!(
            "F32Resampler(selected_backend={:?})",
            self.inner.selected_backend().name()
        )
    }
}

#[pyclass(name = "I16Resampler", module = "fast_audio_resampler", unsendable)]
#[derive(Debug)]
struct PyI16Resampler {
    inner: Resampler<i16>,
    channels: usize,
}

#[pymethods]
impl PyI16Resampler {
    #[new]
    #[pyo3(signature = (
        input_rate,
        output_rate,
        channels,
        quality = None,
        backend = None,
        max_input_frames_per_chunk = None
    ))]
    fn new(
        input_rate: u32,
        output_rate: u32,
        channels: usize,
        quality: Option<&str>,
        backend: Option<&str>,
        max_input_frames_per_chunk: Option<usize>,
    ) -> PyResult<Self> {
        let config = config_from_parts(
            input_rate,
            output_rate,
            channels,
            quality,
            backend,
            max_input_frames_per_chunk,
        )?;

        Ok(Self {
            inner: Resampler::<i16>::new(config).map_err(to_py_err)?,
            channels,
        })
    }

    /// Process interleaved i16 samples and return `(output, stats)`.
    fn process(&mut self, input: Vec<i16>) -> PyResult<(Vec<i16>, PyProcessStats)> {
        let input_frames = input.len() / self.channels;
        let mut output = Vec::with_capacity(self.inner.required_output_capacity(input_frames));
        let stats = self
            .inner
            .process(&input, &mut output)
            .map_err(to_py_err)?
            .into();
        Ok((output, stats))
    }

    /// Finish the stream and return `(output, stats)`.
    fn finish(&mut self) -> PyResult<(Vec<i16>, PyProcessStats)> {
        let mut output = Vec::new();
        let stats = self.inner.finish(&mut output).map_err(to_py_err)?.into();
        Ok((output, stats))
    }

    /// Alias for `finish`.
    fn flush(&mut self) -> PyResult<(Vec<i16>, PyProcessStats)> {
        self.finish()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn required_output_capacity(&self, input_frames: usize) -> usize {
        self.inner.required_output_capacity(input_frames)
    }

    #[getter]
    fn selected_backend(&self) -> &'static str {
        self.inner.selected_backend().name()
    }

    fn __repr__(&self) -> String {
        format!(
            "I16Resampler(selected_backend={:?})",
            self.inner.selected_backend().name()
        )
    }
}

#[pyfunction]
fn quality_values() -> Vec<&'static str> {
    QUALITY_VALUES.to_vec()
}

#[pyfunction]
fn backend_values() -> Vec<&'static str> {
    BACKEND_VALUES.to_vec()
}

#[pymodule]
fn fast_audio_resampler(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyF32Resampler>()?;
    module.add_class::<PyI16Resampler>()?;
    module.add_class::<PyProcessStats>()?;
    module.add_function(wrap_pyfunction!(quality_values, module)?)?;
    module.add_function(wrap_pyfunction!(backend_values, module)?)?;
    Ok(())
}

fn config_from_parts(
    input_rate: u32,
    output_rate: u32,
    channels: usize,
    quality: Option<&str>,
    backend: Option<&str>,
    max_input_frames_per_chunk: Option<usize>,
) -> PyResult<ResamplerConfig> {
    Ok(ResamplerConfig {
        input_rate,
        output_rate,
        channels,
        quality: parse_quality(quality.unwrap_or("balanced"))?,
        backend: parse_backend(backend.unwrap_or("auto"))?,
        max_input_frames_per_chunk,
    })
}

fn parse_quality(value: &str) -> PyResult<Quality> {
    if value.eq_ignore_ascii_case("fast") {
        Ok(Quality::Fast)
    } else if value.eq_ignore_ascii_case("balanced") {
        Ok(Quality::Balanced)
    } else if value.eq_ignore_ascii_case("best") {
        Ok(Quality::Best)
    } else {
        Err(PyValueError::new_err(format!(
            "unknown quality {value:?}; expected one of: {}",
            QUALITY_VALUES.join(", ")
        )))
    }
}

fn parse_backend(value: &str) -> PyResult<FirBackend> {
    if value.eq_ignore_ascii_case("auto") {
        Ok(FirBackend::Auto)
    } else if value.eq_ignore_ascii_case("scalar") {
        Ok(FirBackend::Scalar)
    } else if value.eq_ignore_ascii_case("avx2") || value.eq_ignore_ascii_case("avx2+fma") {
        Ok(FirBackend::Avx2)
    } else if value.eq_ignore_ascii_case("avx512") || value.eq_ignore_ascii_case("avx512f") {
        Ok(FirBackend::Avx512)
    } else if value.eq_ignore_ascii_case("neon") {
        Ok(FirBackend::Neon)
    } else if value.eq_ignore_ascii_case("rvv") {
        Ok(FirBackend::Rvv)
    } else {
        Err(PyValueError::new_err(format!(
            "unknown backend {value:?}; expected one of: {}",
            BACKEND_VALUES.join(", ")
        )))
    }
}

fn to_py_err(error: Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{parse_backend, parse_quality};
    use crate::{FirBackend, Quality};

    #[test]
    fn parses_quality_names_case_insensitively() {
        assert_eq!(parse_quality("FAST").unwrap(), Quality::Fast);
        assert_eq!(parse_quality("balanced").unwrap(), Quality::Balanced);
        assert_eq!(parse_quality("Best").unwrap(), Quality::Best);
        assert!(parse_quality("studio").is_err());
    }

    #[test]
    fn parses_backend_aliases_case_insensitively() {
        assert_eq!(parse_backend("AUTO").unwrap(), FirBackend::Auto);
        assert_eq!(parse_backend("scalar").unwrap(), FirBackend::Scalar);
        assert_eq!(parse_backend("avx2+fma").unwrap(), FirBackend::Avx2);
        assert_eq!(parse_backend("avx512f").unwrap(), FirBackend::Avx512);
        assert!(parse_backend("gpu").is_err());
    }
}
