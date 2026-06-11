#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyType};

use turbo_graph_core::{
    GraphCandidateExplainedSearchReport, GraphCandidateSearchPlan, GraphExplainedSearchReport,
    GraphMemoryCacheBudget, GraphMemoryCacheStats, GraphMemoryIndex as CoreGraphMemoryIndex,
    GraphRerankedHit, GraphSearchPlan, GraphSearchPreset, GraphSearchTelemetry, GraphViewEdge,
    GraphViewNode, GraphViewTrace, MemoryEdge, MemoryHit, MemoryRecord,
};

fn not_contiguous_err(kind: &str) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{kind} must be C-contiguous; call np.ascontiguousarray(...) first",
    ))
}

fn graph_err(err: turbo_graph_core::GraphMemoryError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn graph_preset_from_str(preset: &str) -> PyResult<GraphSearchPreset> {
    match preset {
        "low_latency" => Ok(GraphSearchPreset::low_latency()),
        "balanced" => Ok(GraphSearchPreset::balanced()),
        "broad" => Ok(GraphSearchPreset::broad()),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown graph search preset {other:?}; expected 'low_latency', 'balanced', or 'broad'",
        ))),
    }
}

fn extract_memory_record(record: &Bound<'_, PyDict>) -> PyResult<MemoryRecord> {
    let id: u64 = record
        .get_item("id")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("record missing required 'id'"))?
        .extract()?;
    let title: String = record
        .get_item("title")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("record missing required 'title'"))?
        .extract()?;
    let tags: Vec<String> = record
        .get_item("tags")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("record missing required 'tags'"))?
        .extract()?;

    let mut memory = MemoryRecord::new(id, title, tags);
    if let Some(source) = record.get_item("source")? {
        if !source.is_none() {
            memory = memory.with_source(source.extract::<String>()?);
        }
    }
    if let Some(timestamp_ms) = record.get_item("timestamp_ms")? {
        if !timestamp_ms.is_none() {
            memory = memory.with_timestamp_ms(timestamp_ms.extract::<i64>()?);
        }
    }
    Ok(memory)
}

fn extract_strs<'py>(values: Option<&Bound<'py, PyAny>>, kind: &str) -> PyResult<Vec<String>> {
    match values {
        Some(v) => v.extract::<Vec<String>>().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!("{kind} must be a sequence of strings"))
        }),
        None => Ok(Vec::new()),
    }
}

fn hit_to_dict<'py>(py: Python<'py>, hit: &MemoryHit) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", hit.id)?;
    dict.set_item("score", hit.score)?;
    dict.set_item("title", &hit.title)?;
    dict.set_item("tags", hit.tags.clone())?;
    dict.set_item("source", hit.source.clone())?;
    dict.set_item("timestamp_ms", hit.timestamp_ms)?;
    Ok(dict)
}

fn record_to_dict<'py>(py: Python<'py>, record: &MemoryRecord) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", record.id)?;
    dict.set_item("title", &record.title)?;
    dict.set_item("tags", record.tags.clone())?;
    dict.set_item("source", record.source.clone())?;
    dict.set_item("timestamp_ms", record.timestamp_ms)?;
    Ok(dict)
}

fn edge_to_public_dict<'py>(py: Python<'py>, edge: &MemoryEdge) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("to", edge.to)?;
    dict.set_item("weight", edge.weight)?;
    Ok(dict)
}

fn reranked_hit_to_dict<'py>(
    py: Python<'py>,
    hit: &GraphRerankedHit,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", hit.id)?;
    dict.set_item("score", hit.score)?;
    dict.set_item("vector_score", hit.vector_score)?;
    dict.set_item("graph_score", hit.graph_score)?;
    dict.set_item("depth", hit.depth)?;
    dict.set_item("parent", hit.parent)?;
    dict.set_item("title", &hit.title)?;
    dict.set_item("tags", hit.tags.clone())?;
    dict.set_item("source", hit.source.clone())?;
    dict.set_item("timestamp_ms", hit.timestamp_ms)?;
    Ok(dict)
}

fn plan_to_dict<'py>(py: Python<'py>, plan: GraphSearchPlan) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("total_slots", plan.total_slots)?;
    dict.set_item("graph_slots", plan.graph_slots)?;
    dict.set_item("selected_slots", plan.selected_slots)?;
    dict.set_item("active_blocks", plan.active_blocks)?;
    dict.set_item("graph_cache_hit", plan.graph_cache_hit)?;
    dict.set_item("combined_cache_hit", plan.combined_cache_hit)?;
    dict.set_item("selectivity", plan.selectivity())?;
    dict.set_item("graph_selectivity", plan.graph_selectivity())?;
    dict.set_item("active_block_selectivity", plan.active_block_selectivity())?;
    Ok(dict)
}

fn candidate_plan_to_dict<'py>(
    py: Python<'py>,
    plan: GraphCandidateSearchPlan,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("total_slots", plan.total_slots)?;
    dict.set_item("graph_slots", plan.graph_slots)?;
    dict.set_item("metadata_slots", plan.metadata_slots)?;
    dict.set_item("candidate_input_ids", plan.candidate_input_ids)?;
    dict.set_item("candidate_slots", plan.candidate_slots)?;
    dict.set_item("candidate_missing_ids", plan.candidate_missing_ids)?;
    dict.set_item("candidate_duplicate_ids", plan.candidate_duplicate_ids)?;
    dict.set_item("selected_slots", plan.selected_slots)?;
    dict.set_item("active_blocks", plan.active_blocks)?;
    dict.set_item("graph_cache_hit", plan.graph_cache_hit)?;
    dict.set_item("combined_cache_hit", plan.combined_cache_hit)?;
    dict.set_item("selectivity", plan.selectivity())?;
    dict.set_item("graph_selectivity", plan.graph_selectivity())?;
    dict.set_item("metadata_selectivity", plan.metadata_selectivity())?;
    dict.set_item("candidate_selectivity", plan.candidate_selectivity())?;
    dict.set_item("candidate_live_ratio", plan.candidate_live_ratio())?;
    dict.set_item("candidate_missing_ratio", plan.candidate_missing_ratio())?;
    dict.set_item(
        "candidate_duplicate_ratio",
        plan.candidate_duplicate_ratio(),
    )?;
    dict.set_item("active_block_selectivity", plan.active_block_selectivity())?;
    Ok(dict)
}

fn telemetry_to_dict<'py>(
    py: Python<'py>,
    telemetry: GraphSearchTelemetry,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("view_build_ns", telemetry.view_build_ns)?;
    dict.set_item("vector_search_ns", telemetry.vector_search_ns)?;
    dict.set_item("rerank_ns", telemetry.rerank_ns)?;
    dict.set_item("trace_build_ns", telemetry.trace_build_ns)?;
    dict.set_item("total_ns", telemetry.total_ns)?;
    dict.set_item("blocks_skipped_by_mask", telemetry.blocks_skipped_by_mask)?;
    Ok(dict)
}

fn node_to_dict<'py>(py: Python<'py>, node: &GraphViewNode) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("id", node.id)?;
    dict.set_item("slot", node.slot)?;
    dict.set_item("depth", node.depth)?;
    dict.set_item("parent", node.parent)?;
    dict.set_item("via_weight", node.via_weight)?;
    dict.set_item("path_weight", node.path_weight)?;
    dict.set_item("title", &node.title)?;
    dict.set_item("tags", node.tags.clone())?;
    dict.set_item("source", node.source.clone())?;
    dict.set_item("timestamp_ms", node.timestamp_ms)?;
    Ok(dict)
}

fn edge_to_dict<'py>(py: Python<'py>, edge: &GraphViewEdge) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("from", edge.from)?;
    dict.set_item("to", edge.to)?;
    dict.set_item("weight", edge.weight)?;
    Ok(dict)
}

fn trace_to_dict<'py>(py: Python<'py>, trace: &GraphViewTrace) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("seeds", trace.seeds.clone())?;
    dict.set_item("max_hops", trace.max_hops)?;

    let stats = PyDict::new(py);
    stats.set_item("total_slots", trace.stats.total_slots)?;
    stats.set_item("selected_slots", trace.stats.selected_slots)?;
    stats.set_item("cache_hit", trace.stats.cache_hit)?;
    stats.set_item("selectivity", trace.stats.selectivity())?;
    dict.set_item("stats", stats)?;

    let nodes = PyList::empty(py);
    for node in &trace.nodes {
        nodes.append(node_to_dict(py, node)?)?;
    }
    dict.set_item("nodes", nodes)?;

    let edges = PyList::empty(py);
    for edge in &trace.edges {
        edges.append(edge_to_dict(py, edge)?)?;
    }
    dict.set_item("edges", edges)?;
    Ok(dict)
}

fn cache_stats_to_dict<'py>(
    py: Python<'py>,
    stats: GraphMemoryCacheStats,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("graph_views", stats.graph_views)?;
    dict.set_item("policy_visits", stats.policy_visits)?;
    dict.set_item("policy_views", stats.policy_views)?;
    dict.set_item("combined_views", stats.combined_views)?;
    dict.set_item("combined_policy_views", stats.combined_policy_views)?;
    dict.set_item("tag_masks", stats.tag_masks)?;
    dict.set_item("source_masks", stats.source_masks)?;
    dict.set_item("time_masks", stats.time_masks)?;
    dict.set_item("query_entries", stats.query_entries())?;
    dict.set_item("metadata_entries", stats.metadata_entries())?;
    dict.set_item("total_entries", stats.total_entries)?;
    dict.set_item("query_cache_hits", stats.query_cache_hits())?;
    dict.set_item("query_cache_misses", stats.query_cache_misses())?;
    dict.set_item("metadata_cache_hits", stats.metadata_cache_hits())?;
    dict.set_item("metadata_cache_misses", stats.metadata_cache_misses())?;
    dict.set_item("cache_accesses", stats.cache_accesses())?;
    dict.set_item("cache_hit_ratio", stats.cache_hit_ratio())?;
    dict.set_item("cache_miss_ratio", stats.cache_miss_ratio())?;
    dict.set_item("graph_view_hits", stats.graph_view_hits)?;
    dict.set_item("graph_view_misses", stats.graph_view_misses)?;
    dict.set_item("combined_view_hits", stats.combined_view_hits)?;
    dict.set_item("combined_view_misses", stats.combined_view_misses)?;
    dict.set_item("tag_mask_hits", stats.tag_mask_hits)?;
    dict.set_item("tag_mask_misses", stats.tag_mask_misses)?;
    dict.set_item("source_mask_hits", stats.source_mask_hits)?;
    dict.set_item("source_mask_misses", stats.source_mask_misses)?;
    dict.set_item("time_mask_hits", stats.time_mask_hits)?;
    dict.set_item("time_mask_misses", stats.time_mask_misses)?;
    Ok(dict)
}

fn cache_budget_to_dict<'py>(
    py: Python<'py>,
    budget: GraphMemoryCacheBudget,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("graph_views", budget.graph_views)?;
    dict.set_item("policy_visits", budget.policy_visits)?;
    dict.set_item("policy_views", budget.policy_views)?;
    dict.set_item("combined_views", budget.combined_views)?;
    dict.set_item("combined_policy_views", budget.combined_policy_views)?;
    dict.set_item("tag_masks", budget.tag_masks)?;
    dict.set_item("source_masks", budget.source_masks)?;
    dict.set_item("time_masks", budget.time_masks)?;
    dict.set_item("query_entries", budget.query_entries())?;
    dict.set_item("metadata_entries", budget.metadata_entries())?;
    dict.set_item("total_entries", budget.total_entries())?;
    Ok(dict)
}

fn explained_report_to_dict<'py>(
    py: Python<'py>,
    report: &GraphExplainedSearchReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    let hits = PyList::empty(py);
    for hit in &report.hits {
        hits.append(reranked_hit_to_dict(py, hit)?)?;
    }
    dict.set_item("hits", hits)?;
    dict.set_item("plan", plan_to_dict(py, report.plan)?)?;
    dict.set_item("prefetch_k", report.prefetch_k)?;
    dict.set_item("telemetry", telemetry_to_dict(py, report.telemetry)?)?;
    dict.set_item("trace", trace_to_dict(py, &report.trace)?)?;
    Ok(dict)
}

fn candidate_explained_report_to_dict<'py>(
    py: Python<'py>,
    report: &GraphCandidateExplainedSearchReport,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);

    let hits = PyList::empty(py);
    for hit in &report.hits {
        hits.append(reranked_hit_to_dict(py, hit)?)?;
    }
    dict.set_item("hits", hits)?;
    dict.set_item("plan", candidate_plan_to_dict(py, report.plan)?)?;
    dict.set_item("prefetch_k", report.prefetch_k)?;
    dict.set_item("telemetry", telemetry_to_dict(py, report.telemetry)?)?;
    dict.set_item("trace", trace_to_dict(py, &report.trace)?)?;
    Ok(dict)
}

#[pyclass]
struct TurboQuantIndex {
    inner: turbo_graph_core::TurboQuantIndex,
}

#[pymethods]
impl TurboQuantIndex {
    /// Construct an index. `dim` is optional: when omitted, the
    /// underlying quantized index is created lazily on the first
    /// `add` call, picking up the dimensionality from the input
    /// array's shape.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4))]
    fn new(dim: Option<usize>, bit_width: usize) -> PyResult<Self> {
        let inner = match dim {
            Some(d) => turbo_graph_core::TurboQuantIndex::new(d, bit_width),
            None => turbo_graph_core::TurboQuantIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn add(&mut self, py: Python<'_>, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        let dim = arr.ncols();
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("vectors"))?;
        let vectors = slice.to_vec();
        // `add_2d` handles both eager (dim must match) and lazy (locks
        // dim on first call) cases.
        py.detach(|| self.inner.add_2d(&vectors, dim))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Run a top-`k` search against the index.
    ///
    /// `mask`, when given, is a bool array of length `len(self)`. Only slots
    /// with `mask[i] == True` contribute to the returned top-`k`. The
    /// returned result count per query is `min(k, mask.sum())`.
    #[pyo3(signature = (queries, k, *, mask=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        mask: Option<PyReadonlyArray1<bool>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        let queries = q_slice.to_vec();
        // Reject wrong-dim queries cleanly. Previously the inner
        // `assert_eq!(queries.len(), nq * dim)` would fire as a Rust
        // panic and surface to Python as a PanicException, not the
        // ValueError users expect for input-shape mismatch.
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }

        let mask_arr = mask.as_ref().map(|m| m.as_array());
        let mask_vec: Option<Vec<bool>> = match mask_arr.as_ref() {
            Some(m_arr) => {
                let expected = self.inner.len();
                if m_arr.len() != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "mask length {} does not match index size {}",
                        m_arr.len(),
                        expected,
                    )));
                }
                Some(
                    m_arr
                        .as_slice()
                        .ok_or_else(|| not_contiguous_err("mask"))?
                        .to_vec(),
                )
            }
            None => None,
        };

        let results = py.detach(|| {
            self.inner
                .search_with_mask(&queries, k, mask_vec.as_deref())
        });
        let effective_k = results.k;

        let scores = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), results.indices)
            .unwrap()
            .into_pyarray(py);

        Ok((scores, indices))
    }

    fn write(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.write(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbo_graph_core::TurboQuantIndex::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    /// Warm up the search caches (rotation matrix, Lloyd-Max centroids,
    /// SIMD-blocked code layout) so the first `search` call does not pay
    /// the one-time initialisation cost.
    fn prepare(&self, py: Python<'_>) {
        py.detach(|| self.inner.prepare());
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    ///
    /// The last vector moves into the deleted slot — order is not
    /// preserved. Returns the old index of the moved vector; equals `idx`
    /// when `idx` was already the last element.
    ///
    /// Raises ``IndexError`` if ``idx`` is out of range.
    fn swap_remove(&mut self, idx: usize) -> PyResult<usize> {
        let len = self.inner.len();
        if idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {idx} out of range for index of length {len}",
            )));
        }
        Ok(self.inner.swap_remove(idx))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbo_graph.TurboQuantIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily (no ``dim=``) and hasn't seen an add yet;
    /// otherwise an ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pyclass]
struct IdMapIndex {
    inner: turbo_graph_core::IdMapIndex,
}

#[pymethods]
impl IdMapIndex {
    /// Construct an id-mapped index. `dim` is optional: when omitted,
    /// the underlying quantized index is created lazily on the first
    /// `add_with_ids` call, picking up dim from the input array shape.
    #[new]
    #[pyo3(signature = (dim=None, bit_width=4))]
    fn new(dim: Option<usize>, bit_width: usize) -> PyResult<Self> {
        let inner = match dim {
            Some(d) => turbo_graph_core::IdMapIndex::new(d, bit_width),
            None => turbo_graph_core::IdMapIndex::new_lazy(bit_width),
        }
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Add `n = vectors.shape[0]` vectors with the given external `ids`.
    ///
    /// `ids` must be a 1-D array of `uint64` with length equal to
    /// `vectors.shape[0]`. Raises `ValueError` if any id is already
    /// present or if the lengths don't match. On a lazy index, this
    /// call commits the dimensionality from `vectors.shape[1]`.
    fn add_with_ids(
        &mut self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<u64>,
    ) -> PyResult<()> {
        let v = vectors.as_array();
        let dim = v.ncols();
        let v_slice = v.as_slice().ok_or_else(|| not_contiguous_err("vectors"))?;
        let vectors = v_slice.to_vec();
        let i = ids.as_array();
        let i_slice = i.as_slice().ok_or_else(|| not_contiguous_err("ids"))?;
        let ids = i_slice.to_vec();
        py.detach(|| self.inner.add_with_ids_2d(&vectors, dim, &ids))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Remove the vector with external id `id`. Returns `True` if it was
    /// present, `False` otherwise.
    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Search for the top-`k` nearest external ids for each query.
    ///
    /// `allowlist`, when given, is a `uint64` array of external ids; the
    /// returned top-`k` is restricted to ids in this list. The returned
    /// result count per query is `min(k, len(allowlist))` (after
    /// de-duplication).
    ///
    /// Returns `(scores, ids)` as `(nq, effective_k)` arrays, `ids` typed
    /// `uint64`. Raises `ValueError` for an empty allowlist and `KeyError`
    /// if any allowlist id is not present in the index.
    #[pyo3(signature = (queries, k, *, allowlist=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        allowlist: Option<PyReadonlyArray1<u64>>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<u64>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        let queries = q_slice.to_vec();
        if let Some(idx_dim) = self.inner.dim_opt() {
            if arr.ncols() != idx_dim {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "query dim {} does not match index dim {}",
                    arr.ncols(),
                    idx_dim,
                )));
            }
        }

        let allow_arr = allowlist.as_ref().map(|a| a.as_array());
        let allow_vec: Option<Vec<u64>> = match allow_arr.as_ref() {
            Some(a_arr) => {
                if a_arr.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "allowlist is empty",
                    ));
                }
                let slice = a_arr
                    .as_slice()
                    .ok_or_else(|| not_contiguous_err("allowlist"))?;
                let mut unknown: Vec<u64> = Vec::new();
                for &id in slice {
                    if !self.inner.contains(id) {
                        if unknown.len() < 5 {
                            unknown.push(id);
                        } else {
                            unknown.push(id);
                            break;
                        }
                    }
                }
                if !unknown.is_empty() {
                    let preview: Vec<u64> = unknown.iter().take(5).copied().collect();
                    return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                        "allowlist contains id(s) not present in index: {:?}{}",
                        preview,
                        if unknown.len() > 5 { ", ..." } else { "" },
                    )));
                }
                Some(slice.to_vec())
            }
            None => None,
        };

        let (scores, ids) = py.detach(|| {
            self.inner
                .search_with_allowlist(&queries, k, allow_vec.as_deref())
        });
        // For empty queries (nq=0), match TurboQuantIndex's shape
        // contract: effective_k is `min(k, n_vectors, n_allowed)`. The
        // kernel dedups the allowlist via a packed bool mask for nq>0,
        // so we have to dedup here too — otherwise `allowlist=[1, 1, 1]`
        // returns shape `(0, 3)` for empty queries but `(N, 1)` for
        // non-empty queries, a silent shape divergence.
        let effective_k = if nq == 0 {
            let n_allowed = match allow_vec {
                Some(ref s) => {
                    let mut seen: std::collections::HashSet<u64> =
                        std::collections::HashSet::with_capacity(s.len());
                    s.iter().filter(|id| seen.insert(**id)).count()
                }
                None => self.inner.len(),
            };
            k.min(self.inner.len()).min(n_allowed)
        } else {
            scores.len().checked_div(nq).unwrap_or(0)
        };

        let scores_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), scores)
            .unwrap()
            .into_pyarray(py);
        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, effective_k), ids)
            .unwrap()
            .into_pyarray(py);
        Ok((scores_arr, ids_arr))
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn prepare(&self, py: Python<'_>) {
        py.detach(|| self.inner.prepare());
    }

    /// Serialize the index and id-map side-tables to a `.tvim` file.
    fn write(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.inner.write(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))
    }

    /// Load an `IdMapIndex` from a `.tvim` file previously written by
    /// [`IdMapIndex.write`].
    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = turbo_graph_core::IdMapIndex::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let dim = self
            .inner
            .dim_opt()
            .map_or_else(|| "None".to_string(), |d| d.to_string());
        format!(
            "turbo_graph.IdMapIndex(dim={}, bit_width={}, n_vectors={})",
            dim,
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    /// Vector dimensionality. Returns ``None`` when the index was
    /// constructed lazily and hasn't seen an add yet; otherwise ``int``.
    #[getter]
    fn dim(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pyclass]
struct GraphMemoryIndex {
    inner: CoreGraphMemoryIndex,
}

#[pymethods]
impl GraphMemoryIndex {
    #[new]
    #[pyo3(signature = (dim, bit_width=4))]
    fn new(dim: usize, bit_width: usize) -> PyResult<Self> {
        let inner = CoreGraphMemoryIndex::new(dim, bit_width).map_err(graph_err)?;
        Ok(Self { inner })
    }

    /// Add records from a 2-D float32 matrix plus a sequence of dicts.
    ///
    /// Each record dict requires `id`, `title`, and `tags`; `source` and
    /// `timestamp_ms` are optional.
    fn add_records(
        &mut self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<f32>,
        records: Vec<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let arr = vectors.as_array();
        if arr.ncols() != self.inner.dim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vectors dim {} does not match index dim {}",
                arr.ncols(),
                self.inner.dim(),
            )));
        }
        if arr.nrows() != records.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "records length {} does not match vector rows {}",
                records.len(),
                arr.nrows(),
            )));
        }
        let slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("vectors"))?;
        let vectors = slice.to_vec();
        let parsed = records
            .iter()
            .map(extract_memory_record)
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(|| self.inner.add_records(&vectors, parsed))
            .map_err(graph_err)
    }

    #[pyo3(signature = (id, title, vector, tags, *, source=None, timestamp_ms=None))]
    fn add_node(
        &mut self,
        py: Python<'_>,
        id: u64,
        title: String,
        vector: PyReadonlyArray1<f32>,
        tags: Vec<String>,
        source: Option<String>,
        timestamp_ms: Option<i64>,
    ) -> PyResult<()> {
        let arr = vector.as_array();
        if arr.len() != self.inner.dim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "vector dim {} does not match index dim {}",
                arr.len(),
                self.inner.dim(),
            )));
        }
        let slice = arr.as_slice().ok_or_else(|| not_contiguous_err("vector"))?;
        let vector = slice.to_vec();
        let mut record = MemoryRecord::new(id, title, tags);
        if let Some(source) = source {
            record = record.with_source(source);
        }
        if let Some(timestamp_ms) = timestamp_ms {
            record = record.with_timestamp_ms(timestamp_ms);
        }
        py.detach(|| self.inner.add_records(&vector, vec![record]))
            .map_err(graph_err)
    }

    fn link_directed(&mut self, from_id: u64, to_id: u64, weight: f32) -> PyResult<()> {
        self.inner
            .link_directed(from_id, to_id, weight)
            .map_err(graph_err)
    }

    fn link_bidirectional(&mut self, a: u64, b: u64, weight: f32) -> PyResult<()> {
        self.inner
            .link_bidirectional(a, b, weight)
            .map_err(graph_err)
    }

    fn remove_node(&mut self, id: u64) -> bool {
        self.inner.remove_node(id)
    }

    #[pyo3(signature = (
        query,
        k,
        seeds,
        *,
        max_hops=2,
        required_tags=None,
        allowed_sources=None,
        start_ms=None,
        end_ms=None,
        candidate_ids=None
    ))]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        seeds: Vec<u64>,
        max_hops: usize,
        required_tags: Option<&Bound<'py, PyAny>>,
        allowed_sources: Option<&Bound<'py, PyAny>>,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: Option<Vec<u64>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let arr = query.as_array();
        if arr.len() != self.inner.dim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query dim {} does not match index dim {}",
                arr.len(),
                self.inner.dim(),
            )));
        }
        let q_slice = arr.as_slice().ok_or_else(|| not_contiguous_err("query"))?;
        let query = q_slice.to_vec();
        let tags = extract_strs(required_tags, "required_tags")?;
        let sources = extract_strs(allowed_sources, "allowed_sources")?;
        let tag_refs = tags.iter().map(String::as_str).collect::<Vec<_>>();
        let source_refs = sources.iter().map(String::as_str).collect::<Vec<_>>();
        let hits_vec = py.detach(|| match candidate_ids {
            Some(candidate_ids) => {
                self.inner
                    .search_graph_view_with_metadata_candidates_plan(
                        &query,
                        k,
                        &seeds,
                        max_hops,
                        &tag_refs,
                        &source_refs,
                        start_ms,
                        end_ms,
                        &candidate_ids,
                    )
                    .hits
            }
            None => {
                self.inner
                    .search_graph_view_with_metadata(
                        &query,
                        k,
                        &seeds,
                        max_hops,
                        &tag_refs,
                        &source_refs,
                        start_ms,
                        end_ms,
                    )
                    .hits
            }
        });

        let hits = PyList::empty(py);
        for hit in &hits_vec {
            hits.append(hit_to_dict(py, hit)?)?;
        }
        Ok(hits)
    }

    #[pyo3(signature = (
        queries,
        k,
        seeds,
        *,
        max_hops=2,
        required_tags=None,
        allowed_sources=None,
        start_ms=None,
        end_ms=None,
        candidate_ids=None
    ))]
    fn search_batch<'py>(
        &mut self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        seeds: Vec<u64>,
        max_hops: usize,
        required_tags: Option<&Bound<'py, PyAny>>,
        allowed_sources: Option<&Bound<'py, PyAny>>,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: Option<Vec<u64>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let arr = queries.as_array();
        if arr.ncols() != self.inner.dim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query dim {} does not match index dim {}",
                arr.ncols(),
                self.inner.dim(),
            )));
        }
        let q_slice = arr
            .as_slice()
            .ok_or_else(|| not_contiguous_err("queries"))?;
        let queries = q_slice.to_vec();
        let tags = extract_strs(required_tags, "required_tags")?;
        let sources = extract_strs(allowed_sources, "allowed_sources")?;
        let tag_refs = tags.iter().map(String::as_str).collect::<Vec<_>>();
        let source_refs = sources.iter().map(String::as_str).collect::<Vec<_>>();
        let hit_rows = py.detach(|| match candidate_ids {
            Some(candidate_ids) => {
                self.inner
                    .search_graph_view_with_metadata_candidates_batch_plan(
                        &queries,
                        k,
                        &seeds,
                        max_hops,
                        &tag_refs,
                        &source_refs,
                        start_ms,
                        end_ms,
                        &candidate_ids,
                    )
                    .hits
            }
            None => {
                self.inner
                    .search_graph_view_with_metadata_batch_plan(
                        &queries,
                        k,
                        &seeds,
                        max_hops,
                        &tag_refs,
                        &source_refs,
                        start_ms,
                        end_ms,
                    )
                    .hits
            }
        });

        let rows = PyList::empty(py);
        for row in &hit_rows {
            let hits = PyList::empty(py);
            for hit in row {
                hits.append(hit_to_dict(py, hit)?)?;
            }
            rows.append(hits)?;
        }
        Ok(rows)
    }

    #[pyo3(signature = (
        query,
        k,
        seeds,
        *,
        preset="balanced",
        required_tags=None,
        allowed_sources=None,
        start_ms=None,
        end_ms=None,
        candidate_ids=None
    ))]
    fn explain<'py>(
        &mut self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        seeds: Vec<u64>,
        preset: &str,
        required_tags: Option<&Bound<'py, PyAny>>,
        allowed_sources: Option<&Bound<'py, PyAny>>,
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: Option<Vec<u64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let arr = query.as_array();
        if arr.len() != self.inner.dim() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query dim {} does not match index dim {}",
                arr.len(),
                self.inner.dim(),
            )));
        }
        let q_slice = arr.as_slice().ok_or_else(|| not_contiguous_err("query"))?;
        let query = q_slice.to_vec();
        let tags = extract_strs(required_tags, "required_tags")?;
        let sources = extract_strs(allowed_sources, "allowed_sources")?;
        let tag_refs = tags.iter().map(String::as_str).collect::<Vec<_>>();
        let source_refs = sources.iter().map(String::as_str).collect::<Vec<_>>();
        let preset = graph_preset_from_str(preset)?;
        match candidate_ids {
            Some(candidate_ids) => {
                let tuning = preset.tune(self.inner.len(), k);
                let report = py.detach(|| {
                    self.inner
                        .explain_graph_search_with_policy_metadata_candidates_rerank_timed(
                            &query,
                            k,
                            &seeds,
                            tuning.policy,
                            &tag_refs,
                            &source_refs,
                            start_ms,
                            end_ms,
                            &candidate_ids,
                            tuning.rerank,
                        )
                });
                candidate_explained_report_to_dict(py, &report)
            }
            _ => {
                let report = py.detach(|| {
                    self.inner.explain_graph_search_with_preset(
                        &query,
                        k,
                        &seeds,
                        preset,
                        &tag_refs,
                        &source_refs,
                        start_ms,
                        end_ms,
                    )
                });
                explained_report_to_dict(py, &report)
            }
        }
    }

    fn cache_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        cache_stats_to_dict(py, self.inner.cache_stats())
    }

    fn cache_budget_for_preset<'py>(
        &self,
        py: Python<'py>,
        preset: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let preset = graph_preset_from_str(preset)?;
        cache_budget_to_dict(py, self.inner.cache_budget_for_preset(preset))
    }

    fn clear_query_caches(&mut self) {
        self.inner.clear_query_caches();
    }

    fn clear_metadata_caches(&mut self) {
        self.inner.clear_metadata_caches();
    }

    fn clear_all_caches(&mut self) {
        self.inner.clear_all_caches();
    }

    fn trim_query_caches(&mut self, max_entries_per_cache: usize) {
        self.inner.trim_query_caches(max_entries_per_cache);
    }

    fn trim_metadata_caches(&mut self, max_entries_per_cache: usize) {
        self.inner.trim_metadata_caches(max_entries_per_cache);
    }

    fn trim_all_caches(&mut self, max_entries_per_cache: usize) {
        self.inner.trim_all_caches(max_entries_per_cache);
    }

    fn trim_caches_for_preset<'py>(
        &mut self,
        py: Python<'py>,
        preset: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let preset = graph_preset_from_str(preset)?;
        cache_budget_to_dict(py, self.inner.trim_caches_for_preset(preset))
    }

    fn prepare(&self, py: Python<'_>) {
        py.detach(|| self.inner.prepare());
    }

    fn write(&self, py: Python<'_>, index_path: &str, graph_path: &str) -> PyResult<()> {
        py.detach(|| self.inner.write(index_path, graph_path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, index_path: &str, graph_path: &str) -> PyResult<Self> {
        let inner = CoreGraphMemoryIndex::load(index_path, graph_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    fn slot_of(&self, id: u64) -> Option<usize> {
        self.inner.slot_of(id)
    }

    fn record<'py>(&self, py: Python<'py>, id: u64) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .record(id)
            .map(|record| record_to_dict(py, record))
            .transpose()
    }

    fn neighbors<'py>(&self, py: Python<'py>, id: u64) -> PyResult<Bound<'py, PyList>> {
        if !self.inner.contains(id) {
            return Err(pyo3::exceptions::PyKeyError::new_err(format!(
                "memory id {id} is not present"
            )));
        }
        let neighbors = PyList::empty(py);
        for edge in self.inner.neighbors(id) {
            neighbors.append(edge_to_public_dict(py, edge)?)?;
        }
        Ok(neighbors)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "turbo_graph.GraphMemoryIndex(dim={}, bit_width={}, n_records={})",
            self.inner.dim(),
            self.inner.bit_width(),
            self.inner.len()
        )
    }

    fn __contains__(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bit_width(&self) -> usize {
        self.inner.bit_width()
    }
}

#[pymodule]
fn _turbo_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TurboQuantIndex>()?;
    m.add_class::<IdMapIndex>()?;
    m.add_class::<GraphMemoryIndex>()?;
    Ok(())
}
