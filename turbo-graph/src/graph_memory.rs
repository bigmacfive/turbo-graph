//! Graph-scoped local memory on top of [`TurboQuantIndex`].
//!
//! Plain vector search scores the whole index. `GraphMemoryIndex` keeps a
//! graph/metadata sidecar and compiles graph views into cached [`SlotMask`]
//! values, so repeated local-context queries can score only the relevant slots.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::hash::Hash;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use crate::{AddError, ConstructError, SlotMask, TurboQuantIndex, BLOCK};

const GRAPH_MAGIC: &[u8; 4] = b"TVGM";
const GRAPH_VERSION: u8 = 2;

/// Node metadata stored in the graph-memory sidecar.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryRecord {
    pub id: u64,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

impl MemoryRecord {
    pub fn new<T, I, S>(id: u64, title: T, tags: I) -> Self
    where
        T: Into<String>,
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            id,
            title: title.into(),
            tags: tags.into_iter().map(Into::into).collect(),
            source: None,
            timestamp_ms: None,
        }
    }

    pub fn with_source<S>(mut self, source: S) -> Self
    where
        S: Into<String>,
    {
        self.source = Some(source.into());
        self
    }

    pub fn with_timestamp_ms(mut self, timestamp_ms: i64) -> Self {
        self.timestamp_ms = Some(timestamp_ms);
        self
    }
}

/// Weighted graph edge between memory records.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryEdge {
    pub to: u64,
    pub weight: f32,
}

/// Search result with stable memory id and metadata.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MemoryHit {
    pub id: u64,
    pub score: f32,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

/// Search result after blending vector score with graph-path prior.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphRerankedHit {
    pub id: u64,
    pub score: f32,
    pub vector_score: f32,
    pub graph_score: f32,
    pub depth: usize,
    pub parent: Option<u64>,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

/// Search result after blending vector, graph-path, and external candidate scores.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphHybridHit {
    pub id: u64,
    pub score: f32,
    pub vector_score: f32,
    pub graph_score: f32,
    pub candidate_score: f32,
    pub depth: usize,
    pub parent: Option<u64>,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

/// Candidate-score scaling before hybrid reranking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GraphCandidateScoreNormalization {
    None,
    MinMax,
    MaxAbs,
}

/// Metadata about a compiled graph view.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphViewStats {
    pub total_slots: usize,
    pub selected_slots: usize,
    pub cache_hit: bool,
}

impl GraphViewStats {
    pub fn selectivity(&self) -> f32 {
        if self.total_slots == 0 {
            0.0
        } else {
            self.selected_slots as f32 / self.total_slots as f32
        }
    }
}

/// Controls weighted graph-view expansion.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphViewPolicy {
    pub max_hops: usize,
    pub max_nodes: usize,
    pub max_active_blocks: usize,
    pub min_path_weight: f32,
}

impl GraphViewPolicy {
    pub fn new(max_hops: usize) -> Self {
        Self {
            max_hops,
            max_nodes: usize::MAX,
            max_active_blocks: usize::MAX,
            min_path_weight: 0.0,
        }
    }

    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes;
        self
    }

    pub fn with_max_active_blocks(mut self, max_active_blocks: usize) -> Self {
        self.max_active_blocks = max_active_blocks;
        self
    }

    pub fn with_min_path_weight(mut self, min_path_weight: f32) -> Self {
        self.min_path_weight = min_path_weight;
        self
    }
}

/// Controls graph-aware reranking after TurboQuant candidate retrieval.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphRerankConfig {
    pub vector_weight: f32,
    pub graph_weight: f32,
    pub prefetch_factor: usize,
    pub min_prefetch: usize,
}

impl Default for GraphRerankConfig {
    fn default() -> Self {
        Self {
            vector_weight: 1.0,
            graph_weight: 0.15,
            prefetch_factor: 4,
            min_prefetch: 32,
        }
    }
}

impl GraphRerankConfig {
    pub fn new(vector_weight: f32, graph_weight: f32) -> Self {
        Self {
            vector_weight,
            graph_weight,
            ..Self::default()
        }
    }

    pub fn with_prefetch_factor(mut self, prefetch_factor: usize) -> Self {
        self.prefetch_factor = prefetch_factor;
        self
    }

    pub fn with_min_prefetch(mut self, min_prefetch: usize) -> Self {
        self.min_prefetch = min_prefetch;
        self
    }

    fn normalized(self) -> Self {
        Self {
            vector_weight: finite_or(self.vector_weight, 1.0),
            graph_weight: finite_or(self.graph_weight, 0.0),
            prefetch_factor: self.prefetch_factor.max(1),
            min_prefetch: self.min_prefetch,
        }
    }

    fn prefetch_k(self, k: usize, selected_slots: usize) -> usize {
        if k == 0 || selected_slots == 0 {
            return 0;
        }
        let config = self.normalized();
        let factor_prefetch = k.saturating_mul(config.prefetch_factor);
        factor_prefetch
            .max(config.min_prefetch)
            .max(k)
            .min(selected_slots)
    }
}

/// Controls hybrid reranking over TurboQuant, graph, and external candidates.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphHybridRerankConfig {
    pub vector_weight: f32,
    pub graph_weight: f32,
    pub candidate_weight: f32,
    pub candidate_score_normalization: GraphCandidateScoreNormalization,
    pub prefetch_factor: usize,
    pub min_prefetch: usize,
}

impl Default for GraphHybridRerankConfig {
    fn default() -> Self {
        Self {
            vector_weight: 1.0,
            graph_weight: 0.15,
            candidate_weight: 0.25,
            candidate_score_normalization: GraphCandidateScoreNormalization::None,
            prefetch_factor: 4,
            min_prefetch: 32,
        }
    }
}

impl GraphHybridRerankConfig {
    pub fn new(vector_weight: f32, graph_weight: f32, candidate_weight: f32) -> Self {
        Self {
            vector_weight,
            graph_weight,
            candidate_weight,
            ..Self::default()
        }
    }

    pub fn from_graph_rerank(rerank: GraphRerankConfig, candidate_weight: f32) -> Self {
        Self {
            vector_weight: rerank.vector_weight,
            graph_weight: rerank.graph_weight,
            candidate_weight,
            candidate_score_normalization: Self::default().candidate_score_normalization,
            prefetch_factor: rerank.prefetch_factor,
            min_prefetch: rerank.min_prefetch,
        }
    }

    pub fn with_prefetch_factor(mut self, prefetch_factor: usize) -> Self {
        self.prefetch_factor = prefetch_factor;
        self
    }

    pub fn with_min_prefetch(mut self, min_prefetch: usize) -> Self {
        self.min_prefetch = min_prefetch;
        self
    }

    pub fn with_candidate_score_normalization(
        mut self,
        candidate_score_normalization: GraphCandidateScoreNormalization,
    ) -> Self {
        self.candidate_score_normalization = candidate_score_normalization;
        self
    }

    fn normalized(self) -> Self {
        Self {
            vector_weight: finite_or(self.vector_weight, 1.0),
            graph_weight: finite_or(self.graph_weight, 0.0),
            candidate_weight: finite_or(self.candidate_weight, 0.0),
            candidate_score_normalization: self.candidate_score_normalization,
            prefetch_factor: self.prefetch_factor.max(1),
            min_prefetch: self.min_prefetch,
        }
    }

    fn prefetch_k(self, k: usize, selected_slots: usize) -> usize {
        if k == 0 || selected_slots == 0 {
            return 0;
        }
        let config = self.normalized();
        let factor_prefetch = k.saturating_mul(config.prefetch_factor);
        factor_prefetch
            .max(config.min_prefetch)
            .max(k)
            .min(selected_slots)
    }
}

/// Tuned graph view and rerank settings derived from a preset.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchTuning {
    pub policy: GraphViewPolicy,
    pub rerank: GraphRerankConfig,
}

/// Current graph-memory cache sizes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphMemoryCacheStats {
    pub graph_views: usize,
    pub policy_visits: usize,
    pub policy_views: usize,
    pub combined_views: usize,
    pub combined_policy_views: usize,
    pub tag_masks: usize,
    pub source_masks: usize,
    pub time_masks: usize,
    pub total_entries: usize,
    pub graph_view_hits: usize,
    pub graph_view_misses: usize,
    pub policy_visit_hits: usize,
    pub policy_visit_misses: usize,
    pub policy_view_hits: usize,
    pub policy_view_misses: usize,
    pub combined_view_hits: usize,
    pub combined_view_misses: usize,
    pub combined_policy_view_hits: usize,
    pub combined_policy_view_misses: usize,
    pub tag_mask_hits: usize,
    pub tag_mask_misses: usize,
    pub source_mask_hits: usize,
    pub source_mask_misses: usize,
    pub time_mask_hits: usize,
    pub time_mask_misses: usize,
}

impl GraphMemoryCacheStats {
    pub fn query_entries(&self) -> usize {
        self.graph_views
            + self.policy_visits
            + self.policy_views
            + self.combined_views
            + self.combined_policy_views
    }

    pub fn metadata_entries(&self) -> usize {
        self.tag_masks + self.source_masks + self.time_masks
    }

    pub fn is_empty(&self) -> bool {
        self.total_entries == 0
    }

    pub fn query_cache_hits(&self) -> usize {
        self.graph_view_hits
            + self.policy_visit_hits
            + self.policy_view_hits
            + self.combined_view_hits
            + self.combined_policy_view_hits
    }

    pub fn query_cache_misses(&self) -> usize {
        self.graph_view_misses
            + self.policy_visit_misses
            + self.policy_view_misses
            + self.combined_view_misses
            + self.combined_policy_view_misses
    }

    pub fn metadata_cache_hits(&self) -> usize {
        self.tag_mask_hits + self.source_mask_hits + self.time_mask_hits
    }

    pub fn metadata_cache_misses(&self) -> usize {
        self.tag_mask_misses + self.source_mask_misses + self.time_mask_misses
    }

    pub fn query_cache_hit_ratio(&self) -> f32 {
        let hits = self.query_cache_hits();
        let misses = self.query_cache_misses();
        ratio(hits, hits + misses)
    }

    pub fn metadata_cache_hit_ratio(&self) -> f32 {
        let hits = self.metadata_cache_hits();
        let misses = self.metadata_cache_misses();
        ratio(hits, hits + misses)
    }

    pub fn query_accesses(&self) -> usize {
        self.query_cache_hits() + self.query_cache_misses()
    }

    pub fn metadata_accesses(&self) -> usize {
        self.metadata_cache_hits() + self.metadata_cache_misses()
    }

    pub fn cache_accesses(&self) -> usize {
        self.query_accesses() + self.metadata_accesses()
    }

    pub fn query_cache_miss_ratio(&self) -> f32 {
        ratio(self.query_cache_misses(), self.query_accesses())
    }

    pub fn metadata_cache_miss_ratio(&self) -> f32 {
        ratio(self.metadata_cache_misses(), self.metadata_accesses())
    }

    pub fn cache_hit_ratio(&self) -> f32 {
        ratio(
            self.query_cache_hits() + self.metadata_cache_hits(),
            self.cache_accesses(),
        )
    }

    pub fn cache_miss_ratio(&self) -> f32 {
        ratio(
            self.query_cache_misses() + self.metadata_cache_misses(),
            self.cache_accesses(),
        )
    }
}

/// Per-cache entry caps for long-running graph-memory processes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphMemoryCacheBudget {
    pub graph_views: usize,
    pub policy_visits: usize,
    pub policy_views: usize,
    pub combined_views: usize,
    pub combined_policy_views: usize,
    pub tag_masks: usize,
    pub source_masks: usize,
    pub time_masks: usize,
}

impl GraphMemoryCacheBudget {
    pub fn fixed(max_entries_per_cache: usize) -> Self {
        Self {
            graph_views: max_entries_per_cache,
            policy_visits: max_entries_per_cache,
            policy_views: max_entries_per_cache,
            combined_views: max_entries_per_cache,
            combined_policy_views: max_entries_per_cache,
            tag_masks: max_entries_per_cache,
            source_masks: max_entries_per_cache,
            time_masks: max_entries_per_cache,
        }
    }

    pub fn query_entries(&self) -> usize {
        self.graph_views
            + self.policy_visits
            + self.policy_views
            + self.combined_views
            + self.combined_policy_views
    }

    pub fn metadata_entries(&self) -> usize {
        self.tag_masks + self.source_masks + self.time_masks
    }

    pub fn total_entries(&self) -> usize {
        self.query_entries() + self.metadata_entries()
    }
}

/// Search-engine oriented preset for deriving graph/rerank budgets.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchPreset {
    pub max_hops: usize,
    pub target_active_blocks: usize,
    pub nodes_per_active_block: usize,
    pub min_path_weight: f32,
    pub vector_weight: f32,
    pub graph_weight: f32,
    pub prefetch_factor: usize,
    pub min_prefetch: usize,
}

impl Default for GraphSearchPreset {
    fn default() -> Self {
        Self::balanced()
    }
}

impl GraphSearchPreset {
    pub fn low_latency() -> Self {
        Self {
            max_hops: 2,
            target_active_blocks: 2,
            nodes_per_active_block: 16,
            min_path_weight: 0.35,
            vector_weight: 1.0,
            graph_weight: 0.20,
            prefetch_factor: 3,
            min_prefetch: 16,
        }
    }

    pub fn balanced() -> Self {
        Self {
            max_hops: 2,
            target_active_blocks: 8,
            nodes_per_active_block: 24,
            min_path_weight: 0.20,
            vector_weight: 1.0,
            graph_weight: 0.15,
            prefetch_factor: 4,
            min_prefetch: 32,
        }
    }

    pub fn broad() -> Self {
        Self {
            max_hops: 3,
            target_active_blocks: 32,
            nodes_per_active_block: 32,
            min_path_weight: 0.05,
            vector_weight: 1.0,
            graph_weight: 0.10,
            prefetch_factor: 6,
            min_prefetch: 64,
        }
    }

    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops;
        self
    }

    pub fn with_target_active_blocks(mut self, target_active_blocks: usize) -> Self {
        self.target_active_blocks = target_active_blocks;
        self
    }

    pub fn with_nodes_per_active_block(mut self, nodes_per_active_block: usize) -> Self {
        self.nodes_per_active_block = nodes_per_active_block;
        self
    }

    pub fn with_min_path_weight(mut self, min_path_weight: f32) -> Self {
        self.min_path_weight = min_path_weight;
        self
    }

    pub fn with_graph_weight(mut self, graph_weight: f32) -> Self {
        self.graph_weight = graph_weight;
        self
    }

    pub fn with_prefetch_factor(mut self, prefetch_factor: usize) -> Self {
        self.prefetch_factor = prefetch_factor;
        self
    }

    pub fn with_min_prefetch(mut self, min_prefetch: usize) -> Self {
        self.min_prefetch = min_prefetch;
        self
    }

    pub fn tune(self, total_slots: usize, k: usize) -> GraphSearchTuning {
        let total_blocks = total_slots.saturating_add(BLOCK - 1) / BLOCK;
        let active_blocks = if total_blocks == 0 {
            0
        } else {
            self.target_active_blocks.clamp(1, total_blocks)
        };
        let nodes_per_block = self.nodes_per_active_block.max(1);
        let max_nodes = if total_slots == 0 {
            0
        } else {
            active_blocks
                .saturating_mul(nodes_per_block)
                .max(k)
                .min(total_slots)
        };

        GraphSearchTuning {
            policy: GraphViewPolicy::new(self.max_hops)
                .with_max_nodes(max_nodes)
                .with_max_active_blocks(active_blocks)
                .with_min_path_weight(self.min_path_weight),
            rerank: GraphRerankConfig::new(self.vector_weight, self.graph_weight)
                .with_prefetch_factor(self.prefetch_factor)
                .with_min_prefetch(self.min_prefetch),
        }
    }

    pub fn cache_budget(self, total_slots: usize) -> GraphMemoryCacheBudget {
        let total_blocks = total_slots.saturating_add(BLOCK - 1) / BLOCK;
        let active_blocks = if total_blocks == 0 {
            self.target_active_blocks.max(1)
        } else {
            self.target_active_blocks.clamp(1, total_blocks)
        };
        let scale = active_blocks.saturating_mul(self.max_hops.max(1)).max(1);
        let query_base = scale.saturating_mul(16).clamp(16, 4_096);
        let combined_base = query_base.saturating_mul(2).min(8_192);
        let metadata_base = scale.saturating_mul(8).clamp(16, 2_048);

        GraphMemoryCacheBudget {
            graph_views: query_base,
            policy_visits: query_base,
            policy_views: query_base,
            combined_views: combined_base,
            combined_policy_views: combined_base,
            tag_masks: metadata_base,
            source_masks: metadata_base,
            time_masks: metadata_base,
        }
    }
}

/// Search result plus the graph-view telemetry used to produce it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchReport {
    pub hits: Vec<MemoryHit>,
    pub view: GraphViewStats,
}

/// Search-planning telemetry for a graph plus metadata view.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchPlan {
    pub total_slots: usize,
    pub graph_slots: usize,
    pub selected_slots: usize,
    pub active_blocks: usize,
    pub graph_cache_hit: bool,
    pub combined_cache_hit: bool,
}

impl GraphSearchPlan {
    pub fn view_stats(&self) -> GraphViewStats {
        GraphViewStats {
            total_slots: self.total_slots,
            selected_slots: self.selected_slots,
            cache_hit: self.combined_cache_hit || self.graph_cache_hit,
        }
    }

    pub fn selectivity(&self) -> f32 {
        if self.total_slots == 0 {
            0.0
        } else {
            self.selected_slots as f32 / self.total_slots as f32
        }
    }

    pub fn graph_selectivity(&self) -> f32 {
        if self.total_slots == 0 {
            0.0
        } else {
            self.graph_slots as f32 / self.total_slots as f32
        }
    }

    pub fn active_block_selectivity(&self) -> f32 {
        let total_blocks = self.total_slots.saturating_add(BLOCK - 1) / BLOCK;
        if total_blocks == 0 {
            0.0
        } else {
            self.active_blocks as f32 / total_blocks as f32
        }
    }
}

/// Search result plus full graph/metadata planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphPlannedSearchReport {
    pub hits: Vec<MemoryHit>,
    pub plan: GraphSearchPlan,
}

/// Batch search result plus one shared graph/metadata planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphBatchPlannedSearchReport {
    pub hits: Vec<Vec<MemoryHit>>,
    pub plan: GraphSearchPlan,
}

/// Search-planning telemetry for graph+metadata constrained by candidate ids.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateSearchPlan {
    pub total_slots: usize,
    pub graph_slots: usize,
    pub metadata_slots: usize,
    pub candidate_input_ids: usize,
    pub candidate_slots: usize,
    pub candidate_missing_ids: usize,
    pub candidate_duplicate_ids: usize,
    pub selected_slots: usize,
    pub active_blocks: usize,
    pub graph_cache_hit: bool,
    pub combined_cache_hit: bool,
}

impl GraphCandidateSearchPlan {
    pub fn view_stats(&self) -> GraphViewStats {
        GraphViewStats {
            total_slots: self.total_slots,
            selected_slots: self.selected_slots,
            cache_hit: self.combined_cache_hit || self.graph_cache_hit,
        }
    }

    pub fn selectivity(&self) -> f32 {
        ratio(self.selected_slots, self.total_slots)
    }

    pub fn graph_selectivity(&self) -> f32 {
        ratio(self.graph_slots, self.total_slots)
    }

    pub fn metadata_selectivity(&self) -> f32 {
        ratio(self.metadata_slots, self.total_slots)
    }

    pub fn candidate_selectivity(&self) -> f32 {
        ratio(self.candidate_slots, self.total_slots)
    }

    pub fn candidate_live_ratio(&self) -> f32 {
        ratio(self.candidate_slots, self.candidate_input_ids)
    }

    pub fn candidate_missing_ratio(&self) -> f32 {
        ratio(self.candidate_missing_ids, self.candidate_input_ids)
    }

    pub fn candidate_duplicate_ratio(&self) -> f32 {
        ratio(self.candidate_duplicate_ids, self.candidate_input_ids)
    }

    pub fn active_block_selectivity(&self) -> f32 {
        let total_blocks = self.total_slots.saturating_add(BLOCK - 1) / BLOCK;
        ratio(self.active_blocks, total_blocks)
    }
}

/// Search result plus graph, metadata, and candidate-id planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidatePlannedSearchReport {
    pub hits: Vec<MemoryHit>,
    pub plan: GraphCandidateSearchPlan,
}

/// Batch search result plus graph, metadata, and candidate-id planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateBatchPlannedSearchReport {
    pub hits: Vec<Vec<MemoryHit>>,
    pub plan: GraphCandidateSearchPlan,
}

/// Reranked graph search result plus planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphRerankedSearchReport {
    pub hits: Vec<GraphRerankedHit>,
    pub plan: GraphSearchPlan,
    pub prefetch_k: usize,
}

/// Batch reranked graph search result plus one shared planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphBatchRerankedSearchReport {
    pub hits: Vec<Vec<GraphRerankedHit>>,
    pub plan: GraphSearchPlan,
    pub prefetch_k: usize,
}

/// Reranked graph search result plus candidate-id planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateRerankedSearchReport {
    pub hits: Vec<GraphRerankedHit>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
}

/// Batch reranked graph search result plus one shared candidate planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateBatchRerankedSearchReport {
    pub hits: Vec<Vec<GraphRerankedHit>>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
}

/// Wall-clock and kernel telemetry for one graph-memory search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchTelemetry {
    pub view_build_ns: u128,
    pub vector_search_ns: u128,
    pub rerank_ns: u128,
    pub trace_build_ns: u128,
    pub total_ns: u128,
    pub blocks_skipped_by_mask: u64,
}

/// Reranked graph search result plus planning and timing telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphTimedRerankedSearchReport {
    pub hits: Vec<GraphRerankedHit>,
    pub plan: GraphSearchPlan,
    pub prefetch_k: usize,
    pub telemetry: GraphSearchTelemetry,
}

/// Batch reranked graph search result plus planning and timing telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphBatchTimedRerankedSearchReport {
    pub hits: Vec<Vec<GraphRerankedHit>>,
    pub plan: GraphSearchPlan,
    pub prefetch_k: usize,
    pub telemetry: GraphSearchTelemetry,
}

/// Timed reranked graph search result plus candidate-id planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateTimedRerankedSearchReport {
    pub hits: Vec<GraphRerankedHit>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
    pub telemetry: GraphSearchTelemetry,
}

/// Hybrid reranked search result plus candidate-id planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateHybridSearchReport {
    pub hits: Vec<GraphHybridHit>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
}

/// Batch hybrid reranked search result plus one shared candidate planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateBatchHybridSearchReport {
    pub hits: Vec<Vec<GraphHybridHit>>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
}

/// Timed hybrid reranked search result plus candidate-id planning telemetry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateTimedHybridSearchReport {
    pub hits: Vec<GraphHybridHit>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
    pub telemetry: GraphSearchTelemetry,
}

/// Hybrid reranked graph search plus the graph view used to explain it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateHybridExplainedSearchReport {
    pub hits: Vec<GraphHybridHit>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
    pub telemetry: GraphSearchTelemetry,
    pub trace: GraphViewTrace,
}

impl GraphCandidateHybridExplainedSearchReport {
    pub fn debug_snapshot(&self) -> GraphCandidateHybridSearchDebugSnapshot {
        let mut hit_by_id = HashMap::with_capacity(self.hits.len());
        let hits: Vec<GraphHybridSearchDebugHit> = self
            .hits
            .iter()
            .enumerate()
            .map(|(idx, hit)| {
                let rank = idx + 1;
                hit_by_id.insert(hit.id, (rank, hit));
                GraphHybridSearchDebugHit {
                    rank,
                    id: hit.id,
                    score: hit.score,
                    vector_score: hit.vector_score,
                    graph_score: hit.graph_score,
                    candidate_score: hit.candidate_score,
                    depth: hit.depth,
                    parent: hit.parent,
                    title: hit.title.clone(),
                    tags: hit.tags.clone(),
                    source: hit.source.clone(),
                    timestamp_ms: hit.timestamp_ms,
                }
            })
            .collect();
        let nodes = self
            .trace
            .nodes
            .iter()
            .map(|node| {
                let hit = hit_by_id.get(&node.id).copied();
                GraphHybridSearchDebugNode {
                    id: node.id,
                    slot: node.slot,
                    depth: node.depth,
                    parent: node.parent,
                    via_weight: node.via_weight,
                    path_weight: node.path_weight,
                    hit_rank: hit.map(|(rank, _)| rank),
                    score: hit.map(|(_, hit)| hit.score),
                    vector_score: hit.map(|(_, hit)| hit.vector_score),
                    graph_score: hit.map(|(_, hit)| hit.graph_score),
                    candidate_score: hit.map(|(_, hit)| hit.candidate_score),
                    title: node.title.clone(),
                    tags: node.tags.clone(),
                    source: node.source.clone(),
                    timestamp_ms: node.timestamp_ms,
                }
            })
            .collect();
        let edges = self
            .trace
            .edges
            .iter()
            .map(|edge| GraphSearchDebugEdge {
                from: edge.from,
                to: edge.to,
                weight: edge.weight,
            })
            .collect();

        GraphCandidateHybridSearchDebugSnapshot {
            summary: GraphCandidateSearchDebugSummary {
                total_slots: self.plan.total_slots,
                graph_slots: self.plan.graph_slots,
                metadata_slots: self.plan.metadata_slots,
                candidate_input_ids: self.plan.candidate_input_ids,
                candidate_slots: self.plan.candidate_slots,
                candidate_missing_ids: self.plan.candidate_missing_ids,
                candidate_duplicate_ids: self.plan.candidate_duplicate_ids,
                selected_slots: self.plan.selected_slots,
                active_blocks: self.plan.active_blocks,
                graph_cache_hit: self.plan.graph_cache_hit,
                combined_cache_hit: self.plan.combined_cache_hit,
                selectivity: self.plan.selectivity(),
                graph_selectivity: self.plan.graph_selectivity(),
                metadata_selectivity: self.plan.metadata_selectivity(),
                candidate_selectivity: self.plan.candidate_selectivity(),
                active_block_selectivity: self.plan.active_block_selectivity(),
                prefetch_k: self.prefetch_k,
                hit_count: self.hits.len(),
                trace_node_count: self.trace.nodes.len(),
                trace_edge_count: self.trace.edges.len(),
            },
            telemetry: self.telemetry,
            hits,
            nodes,
            edges,
        }
    }
}

/// Candidate-constrained reranked graph search plus the graph view used to explain it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateExplainedSearchReport {
    pub hits: Vec<GraphRerankedHit>,
    pub plan: GraphCandidateSearchPlan,
    pub prefetch_k: usize,
    pub telemetry: GraphSearchTelemetry,
    pub trace: GraphViewTrace,
}

impl GraphCandidateExplainedSearchReport {
    pub fn debug_snapshot(&self) -> GraphCandidateSearchDebugSnapshot {
        let mut hit_by_id = HashMap::with_capacity(self.hits.len());
        let hits: Vec<GraphSearchDebugHit> = self
            .hits
            .iter()
            .enumerate()
            .map(|(idx, hit)| {
                let rank = idx + 1;
                hit_by_id.insert(hit.id, (rank, hit));
                GraphSearchDebugHit {
                    rank,
                    id: hit.id,
                    score: hit.score,
                    vector_score: hit.vector_score,
                    graph_score: hit.graph_score,
                    depth: hit.depth,
                    parent: hit.parent,
                    title: hit.title.clone(),
                    tags: hit.tags.clone(),
                    source: hit.source.clone(),
                    timestamp_ms: hit.timestamp_ms,
                }
            })
            .collect();
        let nodes = self
            .trace
            .nodes
            .iter()
            .map(|node| {
                let hit = hit_by_id.get(&node.id).copied();
                GraphSearchDebugNode {
                    id: node.id,
                    slot: node.slot,
                    depth: node.depth,
                    parent: node.parent,
                    via_weight: node.via_weight,
                    path_weight: node.path_weight,
                    hit_rank: hit.map(|(rank, _)| rank),
                    score: hit.map(|(_, hit)| hit.score),
                    vector_score: hit.map(|(_, hit)| hit.vector_score),
                    graph_score: hit.map(|(_, hit)| hit.graph_score),
                    title: node.title.clone(),
                    tags: node.tags.clone(),
                    source: node.source.clone(),
                    timestamp_ms: node.timestamp_ms,
                }
            })
            .collect();
        let edges = self
            .trace
            .edges
            .iter()
            .map(|edge| GraphSearchDebugEdge {
                from: edge.from,
                to: edge.to,
                weight: edge.weight,
            })
            .collect();

        GraphCandidateSearchDebugSnapshot {
            summary: GraphCandidateSearchDebugSummary {
                total_slots: self.plan.total_slots,
                graph_slots: self.plan.graph_slots,
                metadata_slots: self.plan.metadata_slots,
                candidate_input_ids: self.plan.candidate_input_ids,
                candidate_slots: self.plan.candidate_slots,
                candidate_missing_ids: self.plan.candidate_missing_ids,
                candidate_duplicate_ids: self.plan.candidate_duplicate_ids,
                selected_slots: self.plan.selected_slots,
                active_blocks: self.plan.active_blocks,
                graph_cache_hit: self.plan.graph_cache_hit,
                combined_cache_hit: self.plan.combined_cache_hit,
                selectivity: self.plan.selectivity(),
                graph_selectivity: self.plan.graph_selectivity(),
                metadata_selectivity: self.plan.metadata_selectivity(),
                candidate_selectivity: self.plan.candidate_selectivity(),
                active_block_selectivity: self.plan.active_block_selectivity(),
                prefetch_k: self.prefetch_k,
                hit_count: self.hits.len(),
                trace_node_count: self.trace.nodes.len(),
                trace_edge_count: self.trace.edges.len(),
            },
            telemetry: self.telemetry,
            hits,
            nodes,
            edges,
        }
    }
}

/// Reranked graph search result plus the graph view used to explain it.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphExplainedSearchReport {
    pub hits: Vec<GraphRerankedHit>,
    pub plan: GraphSearchPlan,
    pub prefetch_k: usize,
    pub telemetry: GraphSearchTelemetry,
    pub trace: GraphViewTrace,
}

impl GraphExplainedSearchReport {
    pub fn debug_snapshot(&self) -> GraphSearchDebugSnapshot {
        let mut hit_by_id = HashMap::with_capacity(self.hits.len());
        let hits: Vec<GraphSearchDebugHit> = self
            .hits
            .iter()
            .enumerate()
            .map(|(idx, hit)| {
                let rank = idx + 1;
                hit_by_id.insert(hit.id, (rank, hit));
                GraphSearchDebugHit {
                    rank,
                    id: hit.id,
                    score: hit.score,
                    vector_score: hit.vector_score,
                    graph_score: hit.graph_score,
                    depth: hit.depth,
                    parent: hit.parent,
                    title: hit.title.clone(),
                    tags: hit.tags.clone(),
                    source: hit.source.clone(),
                    timestamp_ms: hit.timestamp_ms,
                }
            })
            .collect();
        let nodes = self
            .trace
            .nodes
            .iter()
            .map(|node| {
                let hit = hit_by_id.get(&node.id).copied();
                GraphSearchDebugNode {
                    id: node.id,
                    slot: node.slot,
                    depth: node.depth,
                    parent: node.parent,
                    via_weight: node.via_weight,
                    path_weight: node.path_weight,
                    hit_rank: hit.map(|(rank, _)| rank),
                    score: hit.map(|(_, hit)| hit.score),
                    vector_score: hit.map(|(_, hit)| hit.vector_score),
                    graph_score: hit.map(|(_, hit)| hit.graph_score),
                    title: node.title.clone(),
                    tags: node.tags.clone(),
                    source: node.source.clone(),
                    timestamp_ms: node.timestamp_ms,
                }
            })
            .collect();
        let edges = self
            .trace
            .edges
            .iter()
            .map(|edge| GraphSearchDebugEdge {
                from: edge.from,
                to: edge.to,
                weight: edge.weight,
            })
            .collect();

        GraphSearchDebugSnapshot {
            summary: GraphSearchDebugSummary {
                total_slots: self.plan.total_slots,
                graph_slots: self.plan.graph_slots,
                selected_slots: self.plan.selected_slots,
                active_blocks: self.plan.active_blocks,
                graph_cache_hit: self.plan.graph_cache_hit,
                combined_cache_hit: self.plan.combined_cache_hit,
                selectivity: self.plan.selectivity(),
                active_block_selectivity: self.plan.active_block_selectivity(),
                prefetch_k: self.prefetch_k,
                hit_count: self.hits.len(),
                trace_node_count: self.trace.nodes.len(),
                trace_edge_count: self.trace.edges.len(),
            },
            telemetry: self.telemetry,
            hits,
            nodes,
            edges,
        }
    }
}

/// Stable, UI-friendly graph search snapshot with hits merged into trace nodes.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchDebugSnapshot {
    pub summary: GraphSearchDebugSummary,
    pub telemetry: GraphSearchTelemetry,
    pub hits: Vec<GraphSearchDebugHit>,
    pub nodes: Vec<GraphSearchDebugNode>,
    pub edges: Vec<GraphSearchDebugEdge>,
}

/// Pre-compiled graph+metadata view for repeated query execution.
#[derive(Clone)]
pub struct GraphPreparedView {
    pub plan: GraphSearchPlan,
    mask: SlotMask,
}

impl GraphPreparedView {
    pub fn search(&self, index: &GraphMemoryIndex, query: &[f32], k: usize) -> Vec<MemoryHit> {
        if k == 0 {
            return Vec::new();
        }
        index.search_with_slot_mask(query, k, &self.mask)
    }

    pub fn search_batch(
        &self,
        index: &GraphMemoryIndex,
        queries: &[f32],
        k: usize,
    ) -> Vec<Vec<MemoryHit>> {
        if k == 0 {
            return vec![Vec::new(); query_count_for_dim(queries, index.dim())];
        }
        index.search_with_slot_mask_batch(queries, k, &self.mask)
    }
}

/// Pre-compiled budgeted graph+metadata view with graph-path scores for rerank.
#[derive(Clone)]
pub struct GraphPreparedPolicyView {
    pub plan: GraphSearchPlan,
    mask: SlotMask,
    path_by_slot: HashMap<usize, WeightedVisit>,
}

impl GraphPreparedPolicyView {
    pub fn search(&self, index: &GraphMemoryIndex, query: &[f32], k: usize) -> Vec<MemoryHit> {
        if k == 0 {
            return Vec::new();
        }
        index.search_with_slot_mask(query, k, &self.mask)
    }

    pub fn search_batch(
        &self,
        index: &GraphMemoryIndex,
        queries: &[f32],
        k: usize,
    ) -> Vec<Vec<MemoryHit>> {
        if k == 0 {
            return vec![Vec::new(); query_count_for_dim(queries, index.dim())];
        }
        index.search_with_slot_mask_batch(queries, k, &self.mask)
    }

    pub fn search_rerank(
        &self,
        index: &GraphMemoryIndex,
        query: &[f32],
        k: usize,
        rerank: GraphRerankConfig,
    ) -> GraphRerankedSearchReport {
        let prefetch_k = rerank.prefetch_k(k, self.plan.selected_slots);
        if prefetch_k == 0 {
            return GraphRerankedSearchReport {
                hits: Vec::new(),
                plan: self.plan,
                prefetch_k,
            };
        }

        let results = index
            .index
            .search_with_slot_mask(query, prefetch_k, &self.mask);
        let mut hits = index.reranked_hits_from_results(&results, &self.path_by_slot, rerank);
        hits.truncate(k);
        GraphRerankedSearchReport {
            hits,
            plan: self.plan,
            prefetch_k,
        }
    }

    pub fn search_rerank_batch(
        &self,
        index: &GraphMemoryIndex,
        queries: &[f32],
        k: usize,
        rerank: GraphRerankConfig,
    ) -> GraphBatchRerankedSearchReport {
        let prefetch_k = rerank.prefetch_k(k, self.plan.selected_slots);
        if prefetch_k == 0 {
            let nq = query_count_for_dim(queries, index.dim());
            return GraphBatchRerankedSearchReport {
                hits: vec![Vec::new(); nq],
                plan: self.plan,
                prefetch_k,
            };
        }

        let results = index
            .index
            .search_with_slot_mask(queries, prefetch_k, &self.mask);
        let hits: Vec<Vec<GraphRerankedHit>> = (0..results.nq)
            .map(|qi| {
                let mut row = index.reranked_hits_from_results_for_query(
                    &results,
                    qi,
                    &self.path_by_slot,
                    rerank,
                );
                row.truncate(k);
                row
            })
            .collect();
        GraphBatchRerankedSearchReport {
            hits,
            plan: self.plan,
            prefetch_k,
        }
    }

    pub fn search_rerank_batch_timed(
        &self,
        index: &GraphMemoryIndex,
        queries: &[f32],
        k: usize,
        rerank: GraphRerankConfig,
    ) -> GraphBatchTimedRerankedSearchReport {
        let total_start = Instant::now();

        let prefetch_k = rerank.prefetch_k(k, self.plan.selected_slots);

        let mut vector_search_ns = 0;
        let mut rerank_ns = 0;
        let mut blocks_skipped_by_mask = 0;
        let nq = query_count_for_dim(queries, index.dim());
        let mut hits = vec![Vec::new(); nq];

        if prefetch_k > 0 {
            let skipped_before = crate::search::blocks_skipped_by_mask();
            let search_start = Instant::now();
            let results = index
                .index
                .search_with_slot_mask(queries, prefetch_k, &self.mask);
            vector_search_ns = search_start.elapsed().as_nanos();
            let skipped_after = crate::search::blocks_skipped_by_mask();
            blocks_skipped_by_mask = skipped_after.saturating_sub(skipped_before);

            let rerank_start = Instant::now();
            hits = (0..results.nq)
                .map(|qi| {
                    let mut row = index.reranked_hits_from_results_for_query(
                        &results,
                        qi,
                        &self.path_by_slot,
                        rerank,
                    );
                    row.truncate(k);
                    row
                })
                .collect();
            rerank_ns = rerank_start.elapsed().as_nanos();
        }

        GraphBatchTimedRerankedSearchReport {
            hits,
            plan: self.plan,
            prefetch_k,
            telemetry: GraphSearchTelemetry {
                view_build_ns: 0,
                vector_search_ns,
                rerank_ns,
                trace_build_ns: 0,
                total_ns: total_start.elapsed().as_nanos(),
                blocks_skipped_by_mask,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchDebugSummary {
    pub total_slots: usize,
    pub graph_slots: usize,
    pub selected_slots: usize,
    pub active_blocks: usize,
    pub graph_cache_hit: bool,
    pub combined_cache_hit: bool,
    pub selectivity: f32,
    pub active_block_selectivity: f32,
    pub prefetch_k: usize,
    pub hit_count: usize,
    pub trace_node_count: usize,
    pub trace_edge_count: usize,
}

/// UI-friendly candidate-constrained graph search snapshot.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateSearchDebugSnapshot {
    pub summary: GraphCandidateSearchDebugSummary,
    pub telemetry: GraphSearchTelemetry,
    pub hits: Vec<GraphSearchDebugHit>,
    pub nodes: Vec<GraphSearchDebugNode>,
    pub edges: Vec<GraphSearchDebugEdge>,
}

/// UI-friendly hybrid candidate graph search snapshot.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateHybridSearchDebugSnapshot {
    pub summary: GraphCandidateSearchDebugSummary,
    pub telemetry: GraphSearchTelemetry,
    pub hits: Vec<GraphHybridSearchDebugHit>,
    pub nodes: Vec<GraphHybridSearchDebugNode>,
    pub edges: Vec<GraphSearchDebugEdge>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphCandidateSearchDebugSummary {
    pub total_slots: usize,
    pub graph_slots: usize,
    pub metadata_slots: usize,
    pub candidate_input_ids: usize,
    pub candidate_slots: usize,
    pub candidate_missing_ids: usize,
    pub candidate_duplicate_ids: usize,
    pub selected_slots: usize,
    pub active_blocks: usize,
    pub graph_cache_hit: bool,
    pub combined_cache_hit: bool,
    pub selectivity: f32,
    pub graph_selectivity: f32,
    pub metadata_selectivity: f32,
    pub candidate_selectivity: f32,
    pub active_block_selectivity: f32,
    pub prefetch_k: usize,
    pub hit_count: usize,
    pub trace_node_count: usize,
    pub trace_edge_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchDebugHit {
    pub rank: usize,
    pub id: u64,
    pub score: f32,
    pub vector_score: f32,
    pub graph_score: f32,
    pub depth: usize,
    pub parent: Option<u64>,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphHybridSearchDebugHit {
    pub rank: usize,
    pub id: u64,
    pub score: f32,
    pub vector_score: f32,
    pub graph_score: f32,
    pub candidate_score: f32,
    pub depth: usize,
    pub parent: Option<u64>,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchDebugNode {
    pub id: u64,
    pub slot: usize,
    pub depth: usize,
    pub parent: Option<u64>,
    pub via_weight: f32,
    pub path_weight: f32,
    pub hit_rank: Option<usize>,
    pub score: Option<f32>,
    pub vector_score: Option<f32>,
    pub graph_score: Option<f32>,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphHybridSearchDebugNode {
    pub id: u64,
    pub slot: usize,
    pub depth: usize,
    pub parent: Option<u64>,
    pub via_weight: f32,
    pub path_weight: f32,
    pub hit_rank: Option<usize>,
    pub score: Option<f32>,
    pub vector_score: Option<f32>,
    pub graph_score: Option<f32>,
    pub candidate_score: Option<f32>,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchDebugEdge {
    pub from: u64,
    pub to: u64,
    pub weight: f32,
}

/// Search result plus a materialized graph-view trace for UI/debug export.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphSearchTraceReport {
    pub hits: Vec<MemoryHit>,
    pub view: GraphViewStats,
    pub trace: GraphViewTrace,
}

/// One node inside an explainable graph view.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphViewNode {
    pub id: u64,
    pub slot: usize,
    pub depth: usize,
    pub parent: Option<u64>,
    pub via_weight: f32,
    pub path_weight: f32,
    pub title: String,
    pub tags: Vec<String>,
    pub source: Option<String>,
    pub timestamp_ms: Option<i64>,
}

/// One edge visible inside an explainable graph view.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphViewEdge {
    pub from: u64,
    pub to: u64,
    pub weight: f32,
}

/// Materialized graph view for UI/debug export.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GraphViewTrace {
    pub seeds: Vec<u64>,
    pub max_hops: usize,
    pub stats: GraphViewStats,
    pub nodes: Vec<GraphViewNode>,
    pub edges: Vec<GraphViewEdge>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct GraphViewKey {
    seeds: Vec<u64>,
    max_hops: usize,
}

impl GraphViewKey {
    fn new(seeds: &[u64], max_hops: usize) -> Self {
        let mut seeds = seeds.to_vec();
        seeds.sort_unstable();
        seeds.dedup();
        Self { seeds, max_hops }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PolicyGraphViewKey {
    seeds: Vec<u64>,
    max_hops: usize,
    max_nodes: usize,
    max_active_blocks: usize,
    min_path_weight_bits: u32,
}

impl PolicyGraphViewKey {
    fn new(seeds: &[u64], policy: GraphViewPolicy) -> Self {
        let mut seeds = seeds.to_vec();
        seeds.sort_unstable();
        seeds.dedup();
        Self {
            seeds,
            max_hops: policy.max_hops,
            max_nodes: policy.max_nodes,
            max_active_blocks: policy.max_active_blocks,
            min_path_weight_bits: normalize_min_path_weight(policy.min_path_weight).to_bits(),
        }
    }

    fn min_path_weight(&self) -> f32 {
        f32::from_bits(self.min_path_weight_bits)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TimeRangeKey {
    start_ms: Option<i64>,
    end_ms: Option<i64>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CombinedViewKey {
    graph: GraphViewKey,
    required_tags: Vec<String>,
    allowed_sources: Vec<String>,
    time: TimeRangeKey,
}

impl CombinedViewKey {
    fn new(
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> Self {
        Self {
            graph: GraphViewKey::new(seeds, max_hops),
            required_tags: normalized_strings(required_tags),
            allowed_sources: normalized_strings(allowed_sources),
            time: TimeRangeKey { start_ms, end_ms },
        }
    }
}

#[derive(Clone, Debug)]
struct CachedCombinedView {
    mask: SlotMask,
    graph_slots: usize,
}

#[derive(Clone, Debug)]
struct CandidateMaskBuild {
    mask: SlotMask,
    input_ids: usize,
    candidate_slots: usize,
    missing_ids: usize,
    duplicate_ids: usize,
}

#[derive(Clone, Debug)]
struct CandidateScoreMaskBuild {
    candidate: CandidateMaskBuild,
    score_by_slot: HashMap<usize, f32>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct CombinedPolicyViewKey {
    graph: PolicyGraphViewKey,
    required_tags: Vec<String>,
    allowed_sources: Vec<String>,
    time: TimeRangeKey,
}

impl CombinedPolicyViewKey {
    fn new(
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> Self {
        Self {
            graph: PolicyGraphViewKey::new(seeds, policy),
            required_tags: normalized_strings(required_tags),
            allowed_sources: normalized_strings(allowed_sources),
            time: TimeRangeKey { start_ms, end_ms },
        }
    }
}

#[derive(Debug)]
pub enum GraphMemoryError {
    Add(AddError),
    Construct(ConstructError),
    MissingId(u64),
    CorruptSidecar(String),
    Io(io::Error),
}

impl fmt::Display for GraphMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add(err) => write!(f, "{err}"),
            Self::Construct(err) => write!(f, "{err}"),
            Self::MissingId(id) => write!(f, "memory id {id} is not present"),
            Self::CorruptSidecar(msg) => write!(f, "corrupt graph-memory sidecar: {msg}"),
            Self::Io(err) => write!(f, "{err}"),
        }
    }
}

impl Error for GraphMemoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Add(err) => Some(err),
            Self::Construct(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::MissingId(_) | Self::CorruptSidecar(_) => None,
        }
    }
}

impl From<AddError> for GraphMemoryError {
    fn from(value: AddError) -> Self {
        Self::Add(value)
    }
}

impl From<ConstructError> for GraphMemoryError {
    fn from(value: ConstructError) -> Self {
        Self::Construct(value)
    }
}

impl From<io::Error> for GraphMemoryError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CacheHits {
    hits: usize,
    misses: usize,
}

impl CacheHits {
    fn record_hit(&mut self) {
        self.hits += 1;
    }

    fn record_miss(&mut self) {
        self.misses += 1;
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GraphMemoryCacheAccess {
    graph_views: CacheHits,
    policy_visits: CacheHits,
    policy_views: CacheHits,
    combined_views: CacheHits,
    combined_policy_views: CacheHits,
    tag_masks: CacheHits,
    source_masks: CacheHits,
    time_masks: CacheHits,
}

/// Local context-memory layer with graph views compiled into packed masks.
pub struct GraphMemoryIndex {
    index: TurboQuantIndex,
    slot_to_id: Vec<u64>,
    id_to_slot: HashMap<u64, usize>,
    records: HashMap<u64, MemoryRecord>,
    edges: HashMap<u64, Vec<MemoryEdge>>,
    tag_to_ids: HashMap<String, Vec<u64>>,
    source_to_ids: HashMap<String, Vec<u64>>,
    time_index: Vec<(i64, u64)>,
    tag_cache: HashMap<String, SlotMask>,
    source_cache: HashMap<String, SlotMask>,
    time_cache: HashMap<TimeRangeKey, SlotMask>,
    view_cache: HashMap<GraphViewKey, SlotMask>,
    policy_visit_cache: HashMap<PolicyGraphViewKey, Vec<WeightedVisit>>,
    policy_view_cache: HashMap<PolicyGraphViewKey, SlotMask>,
    combined_view_cache: HashMap<CombinedViewKey, CachedCombinedView>,
    combined_policy_view_cache: HashMap<CombinedPolicyViewKey, CachedCombinedView>,
    cache_access: GraphMemoryCacheAccess,
}

impl GraphMemoryIndex {
    pub fn new(dim: usize, bit_width: usize) -> Result<Self, GraphMemoryError> {
        Ok(Self {
            index: TurboQuantIndex::new(dim, bit_width)?,
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
            records: HashMap::new(),
            edges: HashMap::new(),
            tag_to_ids: HashMap::new(),
            source_to_ids: HashMap::new(),
            time_index: Vec::new(),
            tag_cache: HashMap::new(),
            source_cache: HashMap::new(),
            time_cache: HashMap::new(),
            view_cache: HashMap::new(),
            policy_visit_cache: HashMap::new(),
            policy_view_cache: HashMap::new(),
            combined_view_cache: HashMap::new(),
            combined_policy_view_cache: HashMap::new(),
            cache_access: GraphMemoryCacheAccess::default(),
        })
    }

    pub fn from_index(index: TurboQuantIndex) -> Result<Self, GraphMemoryError> {
        if !index.is_empty() {
            return Err(GraphMemoryError::CorruptSidecar(
                "from_index requires an empty TurboQuantIndex".to_string(),
            ));
        }
        Ok(Self {
            index,
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
            records: HashMap::new(),
            edges: HashMap::new(),
            tag_to_ids: HashMap::new(),
            source_to_ids: HashMap::new(),
            time_index: Vec::new(),
            tag_cache: HashMap::new(),
            source_cache: HashMap::new(),
            time_cache: HashMap::new(),
            view_cache: HashMap::new(),
            policy_visit_cache: HashMap::new(),
            policy_view_cache: HashMap::new(),
            combined_view_cache: HashMap::new(),
            combined_policy_view_cache: HashMap::new(),
            cache_access: GraphMemoryCacheAccess::default(),
        })
    }

    pub fn add_node<T, I, S>(
        &mut self,
        id: u64,
        title: T,
        embedding: &[f32],
        tags: I,
    ) -> Result<(), GraphMemoryError>
    where
        T: Into<String>,
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.add_records(embedding, vec![MemoryRecord::new(id, title, tags)])
    }

    /// Add a batch of memory records and vectors in one TurboQuant ingest.
    pub fn add_records(
        &mut self,
        vectors: &[f32],
        mut records: Vec<MemoryRecord>,
    ) -> Result<(), GraphMemoryError> {
        let dim = self.dim();
        if dim == 0 || vectors.len() % dim != 0 {
            return Err(GraphMemoryError::Add(
                AddError::VectorBufferNotMultipleOfDim {
                    vectors_len: vectors.len(),
                    dim,
                },
            ));
        }
        let n = vectors.len() / dim;
        if records.len() != n {
            return Err(GraphMemoryError::Add(AddError::IdsCountMismatch {
                expected: n,
                got: records.len(),
            }));
        }

        let mut seen = HashSet::with_capacity(records.len());
        for record in &records {
            if self.id_to_slot.contains_key(&record.id) || !seen.insert(record.id) {
                return Err(GraphMemoryError::Add(AddError::IdAlreadyPresent(record.id)));
            }
        }

        let base_slot = self.slot_to_id.len();
        self.index.add_2d(vectors, dim)?;
        self.slot_to_id.reserve(records.len());
        self.id_to_slot.reserve(records.len());
        for (offset, record) in records.iter_mut().enumerate() {
            record.tags.sort_unstable();
            record.tags.dedup();
            self.id_to_slot.insert(record.id, base_slot + offset);
            self.slot_to_id.push(record.id);
        }
        for record in records {
            self.index_record_metadata(&record);
            self.records.insert(record.id, record);
        }
        self.invalidate_views();
        Ok(())
    }

    /// Replace a node vector without changing its graph edges or slot.
    ///
    /// The new vector is appended first. If validation fails, the existing
    /// node is left untouched. Then `swap_remove(old_slot)` moves the newly
    /// appended vector into the old slot, keeping sidecar slot tables stable.
    pub fn replace_embedding(
        &mut self,
        id: u64,
        embedding: &[f32],
    ) -> Result<(), GraphMemoryError> {
        let Some(&old_slot) = self.id_to_slot.get(&id) else {
            return Err(GraphMemoryError::MissingId(id));
        };
        let dim = self.dim();
        if embedding.len() != dim {
            return Err(GraphMemoryError::Add(
                AddError::VectorBufferNotMultipleOfDim {
                    vectors_len: embedding.len(),
                    dim,
                },
            ));
        }

        let old_len = self.slot_to_id.len();
        self.index.add_2d(embedding, dim)?;
        debug_assert_eq!(self.index.len(), old_len + 1);
        let moved_from = self.index.swap_remove(old_slot);
        debug_assert_eq!(moved_from, old_len);
        debug_assert_eq!(self.index.len(), old_len);
        self.invalidate_views();
        Ok(())
    }

    /// Replace title, tags, source, and timestamp without changing the vector,
    /// slot, or graph edges.
    pub fn replace_record_metadata(
        &mut self,
        mut record: MemoryRecord,
    ) -> Result<(), GraphMemoryError> {
        if !self.id_to_slot.contains_key(&record.id) {
            return Err(GraphMemoryError::MissingId(record.id));
        }
        record.tags.sort_unstable();
        record.tags.dedup();

        let old_record = self
            .records
            .get(&record.id)
            .expect("id table and records are in sync")
            .clone();
        self.unindex_record_metadata(&old_record);
        self.index_record_metadata(&record);
        self.records.insert(record.id, record);
        self.invalidate_metadata_views();
        Ok(())
    }

    pub fn remove_node(&mut self, id: u64) -> bool {
        let Some(slot) = self.id_to_slot.remove(&id) else {
            return false;
        };
        let last = self.slot_to_id.len() - 1;
        let moved_from = self.index.swap_remove(slot);
        debug_assert_eq!(moved_from, last);

        if slot != last {
            let moved_id = self.slot_to_id[last];
            self.slot_to_id[slot] = moved_id;
            self.id_to_slot.insert(moved_id, slot);
        }
        self.slot_to_id.pop();
        if let Some(record) = self.records.remove(&id) {
            self.unindex_record_metadata(&record);
        }
        self.edges.remove(&id);
        for neighbors in self.edges.values_mut() {
            neighbors.retain(|edge| edge.to != id);
        }
        self.invalidate_views();
        true
    }

    pub fn link_directed(
        &mut self,
        from: u64,
        to: u64,
        weight: f32,
    ) -> Result<(), GraphMemoryError> {
        self.require_id(from)?;
        self.require_id(to)?;
        let neighbors = self.edges.entry(from).or_default();
        if let Some(edge) = neighbors.iter_mut().find(|edge| edge.to == to) {
            edge.weight = weight;
        } else {
            neighbors.push(MemoryEdge { to, weight });
        }
        self.invalidate_views();
        Ok(())
    }

    pub fn link_bidirectional(
        &mut self,
        a: u64,
        b: u64,
        weight: f32,
    ) -> Result<(), GraphMemoryError> {
        self.link_directed(a, b, weight)?;
        self.link_directed(b, a, weight)?;
        Ok(())
    }

    pub fn graph_view_mask(&mut self, seeds: &[u64], max_hops: usize) -> SlotMask {
        self.graph_view_mask_with_stats(seeds, max_hops).0
    }

    pub fn graph_view_mask_with_stats(
        &mut self,
        seeds: &[u64],
        max_hops: usize,
    ) -> (SlotMask, GraphViewStats) {
        let key = GraphViewKey::new(seeds, max_hops);
        if let Some(mask) = self.view_cache.get(&key) {
            self.cache_access.graph_views.record_hit();
            return (
                mask.clone(),
                GraphViewStats {
                    total_slots: self.len(),
                    selected_slots: mask.count(),
                    cache_hit: true,
                },
            );
        }

        self.cache_access.graph_views.record_miss();

        let mut mask = SlotMask::new(self.len());
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        for &seed in &key.seeds {
            if self.id_to_slot.contains_key(&seed) && seen.insert(seed) {
                queue.push_back((seed, 0usize));
            }
        }

        while let Some((id, depth)) = queue.pop_front() {
            if let Some(&slot) = self.id_to_slot.get(&id) {
                mask.allow(slot);
            }
            if depth == max_hops {
                continue;
            }
            if let Some(neighbors) = self.edges.get(&id) {
                for edge in neighbors {
                    if seen.insert(edge.to) {
                        queue.push_back((edge.to, depth + 1));
                    }
                }
            }
        }

        self.view_cache.insert(key, mask.clone());
        let stats = GraphViewStats {
            total_slots: self.len(),
            selected_slots: mask.count(),
            cache_hit: false,
        };
        (mask, stats)
    }

    pub fn explain_graph_view(&mut self, seeds: &[u64], max_hops: usize) -> GraphViewTrace {
        let key = GraphViewKey::new(seeds, max_hops);
        let (_mask, stats) = self.graph_view_mask_with_stats(&key.seeds, max_hops);
        self.build_graph_view_trace(key, stats)
    }

    pub fn graph_view_mask_with_policy(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
    ) -> SlotMask {
        self.graph_view_mask_with_policy_stats(seeds, policy).0
    }

    pub fn graph_view_mask_with_policy_stats(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
    ) -> (SlotMask, GraphViewStats) {
        let key = PolicyGraphViewKey::new(seeds, policy);
        if let Some(mask) = self.policy_view_cache.get(&key) {
            self.cache_access.policy_views.record_hit();
            return (
                mask.clone(),
                GraphViewStats {
                    total_slots: self.len(),
                    selected_slots: mask.count(),
                    cache_hit: true,
                },
            );
        }
        self.cache_access.policy_views.record_miss();
        let visits = self.policy_visits(&key);
        let mask = SlotMask::from_slots(
            self.len(),
            visits
                .iter()
                .filter_map(|visit| self.id_to_slot.get(&visit.id).copied()),
        );
        self.policy_view_cache.insert(key, mask.clone());
        let stats = GraphViewStats {
            total_slots: self.len(),
            selected_slots: mask.count(),
            cache_hit: false,
        };
        (mask, stats)
    }

    pub fn explain_graph_view_with_policy(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
    ) -> GraphViewTrace {
        let key = PolicyGraphViewKey::new(seeds, policy);
        let (_mask, stats) = self.graph_view_mask_with_policy_stats(&key.seeds, policy);
        let visits = self.policy_visits(&key);
        self.build_policy_graph_view_trace(key, stats, &visits)
    }

    pub fn tag_mask(&self, tag: &str) -> SlotMask {
        match self.tag_to_ids.get(tag) {
            Some(ids) => SlotMask::from_slots(
                self.len(),
                ids.iter().filter_map(|id| self.id_to_slot.get(id).copied()),
            ),
            None => SlotMask::new(self.len()),
        }
    }

    pub fn tag_view_mask(&mut self, tag: &str) -> SlotMask {
        if let Some(mask) = self.tag_cache.get(tag) {
            self.cache_access.tag_masks.record_hit();
            return mask.clone();
        }
        self.cache_access.tag_masks.record_miss();
        let mask = self.tag_mask(tag);
        self.tag_cache.insert(tag.to_string(), mask.clone());
        mask
    }

    pub fn source_mask(&self, source: &str) -> SlotMask {
        match self.source_to_ids.get(source) {
            Some(ids) => SlotMask::from_slots(
                self.len(),
                ids.iter().filter_map(|id| self.id_to_slot.get(id).copied()),
            ),
            None => SlotMask::new(self.len()),
        }
    }

    pub fn source_view_mask(&mut self, source: &str) -> SlotMask {
        if let Some(mask) = self.source_cache.get(source) {
            self.cache_access.source_masks.record_hit();
            return mask.clone();
        }
        self.cache_access.source_masks.record_miss();
        let mask = self.source_mask(source);
        self.source_cache.insert(source.to_string(), mask.clone());
        mask
    }

    /// Build a timestamp mask over the half-open range `[start_ms, end_ms)`.
    ///
    /// Records without a timestamp are excluded from time-window filters.
    pub fn time_range_mask(&self, start_ms: Option<i64>, end_ms: Option<i64>) -> SlotMask {
        if matches!((start_ms, end_ms), (Some(start), Some(end)) if start >= end) {
            return SlotMask::new(self.len());
        }

        let start_idx = match start_ms {
            Some(start) => self
                .time_index
                .partition_point(|&(timestamp, _)| timestamp < start),
            None => 0,
        };
        let end_idx = match end_ms {
            Some(end) => self
                .time_index
                .partition_point(|&(timestamp, _)| timestamp < end),
            None => self.time_index.len(),
        };

        SlotMask::from_slots(
            self.len(),
            self.time_index[start_idx..end_idx]
                .iter()
                .filter_map(|&(_, id)| self.id_to_slot.get(&id).copied()),
        )
    }

    pub fn time_range_view_mask(&mut self, start_ms: Option<i64>, end_ms: Option<i64>) -> SlotMask {
        let key = TimeRangeKey { start_ms, end_ms };
        if let Some(mask) = self.time_cache.get(&key) {
            self.cache_access.time_masks.record_hit();
            return mask.clone();
        }
        self.cache_access.time_masks.record_miss();
        let mask = self.time_range_mask(start_ms, end_ms);
        self.time_cache.insert(key, mask.clone());
        mask
    }

    /// Build a slot mask from stable memory ids, ignoring ids no longer present.
    pub fn candidate_id_mask(&self, ids: &[u64]) -> SlotMask {
        self.candidate_id_mask_with_stats(ids).mask
    }

    pub fn tuning_for_preset(&self, k: usize, preset: GraphSearchPreset) -> GraphSearchTuning {
        preset.tune(self.len(), k)
    }

    pub fn graph_view_mask_with_metadata_plan(
        &mut self,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> (SlotMask, GraphSearchPlan) {
        let key = CombinedViewKey::new(
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        if let Some(cached) = self.combined_view_cache.get(&key) {
            self.cache_access.combined_views.record_hit();
            let mask = cached.mask.clone();
            let plan = GraphSearchPlan {
                total_slots: self.len(),
                graph_slots: cached.graph_slots,
                selected_slots: mask.count(),
                active_blocks: mask.active_block_count(),
                graph_cache_hit: true,
                combined_cache_hit: true,
            };
            return (mask, plan);
        }
        self.cache_access.combined_views.record_miss();

        let (mut mask, graph_stats) =
            self.graph_view_mask_with_stats(&key.graph.seeds, key.graph.max_hops);
        let required_tags: Vec<&str> = key.required_tags.iter().map(String::as_str).collect();
        let allowed_sources: Vec<&str> = key.allowed_sources.iter().map(String::as_str).collect();
        self.apply_metadata_filters(
            &mut mask,
            &required_tags,
            &allowed_sources,
            key.time.start_ms,
            key.time.end_ms,
        );
        let plan = GraphSearchPlan {
            total_slots: self.len(),
            graph_slots: graph_stats.selected_slots,
            selected_slots: mask.count(),
            active_blocks: mask.active_block_count(),
            graph_cache_hit: graph_stats.cache_hit,
            combined_cache_hit: false,
        };
        self.combined_view_cache.insert(
            key,
            CachedCombinedView {
                mask: mask.clone(),
                graph_slots: plan.graph_slots,
            },
        );
        (mask, plan)
    }

    pub fn graph_view_mask_with_policy_metadata_plan(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> (SlotMask, GraphSearchPlan) {
        let key = CombinedPolicyViewKey::new(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        if let Some(cached) = self.combined_policy_view_cache.get(&key) {
            self.cache_access.combined_policy_views.record_hit();
            let mask = cached.mask.clone();
            let plan = GraphSearchPlan {
                total_slots: self.len(),
                graph_slots: cached.graph_slots,
                selected_slots: mask.count(),
                active_blocks: mask.active_block_count(),
                graph_cache_hit: true,
                combined_cache_hit: true,
            };
            return (mask, plan);
        }
        self.cache_access.combined_policy_views.record_miss();

        let policy = GraphViewPolicy {
            max_hops: key.graph.max_hops,
            max_nodes: key.graph.max_nodes,
            max_active_blocks: key.graph.max_active_blocks,
            min_path_weight: key.graph.min_path_weight(),
        };
        let (mut mask, graph_stats) =
            self.graph_view_mask_with_policy_stats(&key.graph.seeds, policy);
        let required_tags: Vec<&str> = key.required_tags.iter().map(String::as_str).collect();
        let allowed_sources: Vec<&str> = key.allowed_sources.iter().map(String::as_str).collect();
        self.apply_metadata_filters(
            &mut mask,
            &required_tags,
            &allowed_sources,
            key.time.start_ms,
            key.time.end_ms,
        );
        let plan = GraphSearchPlan {
            total_slots: self.len(),
            graph_slots: graph_stats.selected_slots,
            selected_slots: mask.count(),
            active_blocks: mask.active_block_count(),
            graph_cache_hit: graph_stats.cache_hit,
            combined_cache_hit: false,
        };
        self.combined_policy_view_cache.insert(
            key,
            CachedCombinedView {
                mask: mask.clone(),
                graph_slots: plan.graph_slots,
            },
        );
        (mask, plan)
    }

    pub fn graph_view_mask_with_metadata_candidates_plan(
        &mut self,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
    ) -> (SlotMask, GraphCandidateSearchPlan) {
        let (mut mask, base_plan) = self.graph_view_mask_with_metadata_plan(
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let candidate = self.candidate_id_mask_with_stats(candidate_ids);
        mask.intersect_with(&candidate.mask);
        let plan = GraphCandidateSearchPlan {
            total_slots: base_plan.total_slots,
            graph_slots: base_plan.graph_slots,
            metadata_slots: base_plan.selected_slots,
            candidate_input_ids: candidate.input_ids,
            candidate_slots: candidate.candidate_slots,
            candidate_missing_ids: candidate.missing_ids,
            candidate_duplicate_ids: candidate.duplicate_ids,
            selected_slots: mask.count(),
            active_blocks: mask.active_block_count(),
            graph_cache_hit: base_plan.graph_cache_hit,
            combined_cache_hit: base_plan.combined_cache_hit,
        };
        (mask, plan)
    }

    pub fn graph_view_mask_with_policy_metadata_candidates_plan(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
    ) -> (SlotMask, GraphCandidateSearchPlan) {
        let (mut mask, base_plan) = self.graph_view_mask_with_policy_metadata_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let candidate = self.candidate_id_mask_with_stats(candidate_ids);
        mask.intersect_with(&candidate.mask);
        let plan = GraphCandidateSearchPlan {
            total_slots: base_plan.total_slots,
            graph_slots: base_plan.graph_slots,
            metadata_slots: base_plan.selected_slots,
            candidate_input_ids: candidate.input_ids,
            candidate_slots: candidate.candidate_slots,
            candidate_missing_ids: candidate.missing_ids,
            candidate_duplicate_ids: candidate.duplicate_ids,
            selected_slots: mask.count(),
            active_blocks: mask.active_block_count(),
            graph_cache_hit: base_plan.graph_cache_hit,
            combined_cache_hit: base_plan.combined_cache_hit,
        };
        (mask, plan)
    }

    fn graph_view_mask_with_policy_metadata_candidate_scores_plan(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_scores: &[(u64, f32)],
    ) -> (SlotMask, GraphCandidateSearchPlan, HashMap<usize, f32>) {
        let (mut mask, base_plan) = self.graph_view_mask_with_policy_metadata_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let candidate = self.candidate_score_mask_with_stats(candidate_scores);
        mask.intersect_with(&candidate.candidate.mask);
        let plan = GraphCandidateSearchPlan {
            total_slots: base_plan.total_slots,
            graph_slots: base_plan.graph_slots,
            metadata_slots: base_plan.selected_slots,
            candidate_input_ids: candidate.candidate.input_ids,
            candidate_slots: candidate.candidate.candidate_slots,
            candidate_missing_ids: candidate.candidate.missing_ids,
            candidate_duplicate_ids: candidate.candidate.duplicate_ids,
            selected_slots: mask.count(),
            active_blocks: mask.active_block_count(),
            graph_cache_hit: base_plan.graph_cache_hit,
            combined_cache_hit: base_plan.combined_cache_hit,
        };
        (mask, plan, candidate.score_by_slot)
    }

    pub fn search_graph_view(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
    ) -> Vec<MemoryHit> {
        self.search_graph_view_with_stats(query, k, seeds, max_hops, required_tags)
            .hits
    }

    pub fn search_graph_view_with_stats(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
    ) -> GraphSearchReport {
        let planned = self.search_graph_view_with_metadata_plan(
            query,
            k,
            seeds,
            max_hops,
            required_tags,
            &[],
            None,
            None,
        );
        GraphSearchReport {
            hits: planned.hits,
            view: planned.plan.view_stats(),
        }
    }

    pub fn search_graph_view_with_metadata(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphSearchReport {
        let planned = self.search_graph_view_with_metadata_plan(
            query,
            k,
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        GraphSearchReport {
            hits: planned.hits,
            view: planned.plan.view_stats(),
        }
    }

    pub fn search_graph_view_with_metadata_plan(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphPlannedSearchReport {
        let (mask, plan) = self.graph_view_mask_with_metadata_plan(
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let hits = self.search_with_slot_mask(query, k, &mask);
        GraphPlannedSearchReport { hits, plan }
    }

    pub fn search_graph_view_with_metadata_batch_plan(
        &mut self,
        queries: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphBatchPlannedSearchReport {
        let (mask, plan) = self.graph_view_mask_with_metadata_plan(
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let hits = self.search_with_slot_mask_batch(queries, k, &mask);
        GraphBatchPlannedSearchReport { hits, plan }
    }

    pub fn search_graph_view_with_metadata_candidates_plan(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
    ) -> GraphCandidatePlannedSearchReport {
        let (mask, plan) = self.graph_view_mask_with_metadata_candidates_plan(
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_ids,
        );
        let hits = self.search_with_slot_mask(query, k, &mask);
        GraphCandidatePlannedSearchReport { hits, plan }
    }

    pub fn search_graph_view_with_metadata_candidates_batch_plan(
        &mut self,
        queries: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
    ) -> GraphCandidateBatchPlannedSearchReport {
        let (mask, plan) = self.graph_view_mask_with_metadata_candidates_plan(
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_ids,
        );
        let hits = self.search_with_slot_mask_batch(queries, k, &mask);
        GraphCandidateBatchPlannedSearchReport { hits, plan }
    }

    pub fn search_graph_view_with_policy_metadata_plan(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphPlannedSearchReport {
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let hits = self.search_with_slot_mask(query, k, &mask);
        GraphPlannedSearchReport { hits, plan }
    }

    pub fn search_graph_view_with_policy_metadata_candidates_plan(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
    ) -> GraphCandidatePlannedSearchReport {
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_candidates_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_ids,
        );
        let hits = self.search_with_slot_mask(query, k, &mask);
        GraphCandidatePlannedSearchReport { hits, plan }
    }

    pub fn search_graph_view_with_policy_metadata_candidates_rerank(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
        rerank: GraphRerankConfig,
    ) -> GraphCandidateRerankedSearchReport {
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_candidates_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_ids,
        );
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);
        if prefetch_k == 0 {
            return GraphCandidateRerankedSearchReport {
                hits: Vec::new(),
                plan,
                prefetch_k,
            };
        }

        assert_eq!(
            query.len(),
            self.dim(),
            "GraphMemoryIndex search expects a single query vector"
        );
        let results = self.index.search_with_slot_mask(query, prefetch_k, &mask);
        let mut hits = self.reranked_hits_from_results(&results, &path_by_slot, rerank);
        hits.truncate(k);
        GraphCandidateRerankedSearchReport {
            hits,
            plan,
            prefetch_k,
        }
    }

    pub fn search_graph_view_with_policy_metadata_candidates_rerank_batch(
        &mut self,
        queries: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
        rerank: GraphRerankConfig,
    ) -> GraphCandidateBatchRerankedSearchReport {
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_candidates_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_ids,
        );
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);
        if prefetch_k == 0 {
            let nq = query_count_for_dim(queries, self.dim());
            return GraphCandidateBatchRerankedSearchReport {
                hits: vec![Vec::new(); nq],
                plan,
                prefetch_k,
            };
        }

        let results = self.index.search_with_slot_mask(queries, prefetch_k, &mask);
        let hits: Vec<Vec<GraphRerankedHit>> = (0..results.nq)
            .map(|qi| {
                let mut row =
                    self.reranked_hits_from_results_for_query(&results, qi, &path_by_slot, rerank);
                row.truncate(k);
                row
            })
            .collect();
        GraphCandidateBatchRerankedSearchReport {
            hits,
            plan,
            prefetch_k,
        }
    }

    pub fn search_graph_view_with_policy_metadata_candidate_scores_hybrid(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_scores: &[(u64, f32)],
        rerank: GraphHybridRerankConfig,
    ) -> GraphCandidateHybridSearchReport {
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan, mut candidate_score_by_slot) = self
            .graph_view_mask_with_policy_metadata_candidate_scores_plan(
                seeds,
                policy,
                required_tags,
                allowed_sources,
                start_ms,
                end_ms,
                candidate_scores,
            );
        normalize_candidate_score_map(
            &mut candidate_score_by_slot,
            rerank.candidate_score_normalization,
            Some(&mask),
        );
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);
        if prefetch_k == 0 {
            return GraphCandidateHybridSearchReport {
                hits: Vec::new(),
                plan,
                prefetch_k,
            };
        }

        assert_eq!(
            query.len(),
            self.dim(),
            "GraphMemoryIndex search expects a single query vector"
        );
        let results = self.index.search_with_slot_mask(query, prefetch_k, &mask);
        let mut hits = self.hybrid_hits_from_results(
            &results,
            &path_by_slot,
            &candidate_score_by_slot,
            rerank,
        );
        hits.truncate(k);
        GraphCandidateHybridSearchReport {
            hits,
            plan,
            prefetch_k,
        }
    }

    pub fn search_graph_view_with_policy_metadata_candidate_scores_hybrid_batch(
        &mut self,
        queries: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_scores: &[(u64, f32)],
        rerank: GraphHybridRerankConfig,
    ) -> GraphCandidateBatchHybridSearchReport {
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan, mut candidate_score_by_slot) = self
            .graph_view_mask_with_policy_metadata_candidate_scores_plan(
                seeds,
                policy,
                required_tags,
                allowed_sources,
                start_ms,
                end_ms,
                candidate_scores,
            );
        normalize_candidate_score_map(
            &mut candidate_score_by_slot,
            rerank.candidate_score_normalization,
            Some(&mask),
        );
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);
        if prefetch_k == 0 {
            let nq = query_count_for_dim(queries, self.dim());
            return GraphCandidateBatchHybridSearchReport {
                hits: vec![Vec::new(); nq],
                plan,
                prefetch_k,
            };
        }

        let results = self.index.search_with_slot_mask(queries, prefetch_k, &mask);
        let hits: Vec<Vec<GraphHybridHit>> = (0..results.nq)
            .map(|qi| {
                let mut row = self.hybrid_hits_from_results_for_query(
                    &results,
                    qi,
                    &path_by_slot,
                    &candidate_score_by_slot,
                    rerank,
                );
                row.truncate(k);
                row
            })
            .collect();
        GraphCandidateBatchHybridSearchReport {
            hits,
            plan,
            prefetch_k,
        }
    }

    pub fn search_graph_view_with_policy_metadata_rerank(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        rerank: GraphRerankConfig,
    ) -> GraphRerankedSearchReport {
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);
        if prefetch_k == 0 {
            return GraphRerankedSearchReport {
                hits: Vec::new(),
                plan,
                prefetch_k,
            };
        }

        assert_eq!(
            query.len(),
            self.dim(),
            "GraphMemoryIndex search expects a single query vector"
        );
        let results = self.index.search_with_slot_mask(query, prefetch_k, &mask);
        let mut hits = self.reranked_hits_from_results(&results, &path_by_slot, rerank);
        hits.truncate(k);
        GraphRerankedSearchReport {
            hits,
            plan,
            prefetch_k,
        }
    }

    pub fn search_graph_view_with_policy_metadata_rerank_batch(
        &mut self,
        queries: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        rerank: GraphRerankConfig,
    ) -> GraphBatchRerankedSearchReport {
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);
        if prefetch_k == 0 {
            let nq = query_count_for_dim(queries, self.dim());
            return GraphBatchRerankedSearchReport {
                hits: vec![Vec::new(); nq],
                plan,
                prefetch_k,
            };
        }

        let results = self.index.search_with_slot_mask(queries, prefetch_k, &mask);
        let hits: Vec<Vec<GraphRerankedHit>> = (0..results.nq)
            .map(|qi| {
                let mut row =
                    self.reranked_hits_from_results_for_query(&results, qi, &path_by_slot, rerank);
                row.truncate(k);
                row
            })
            .collect();
        GraphBatchRerankedSearchReport {
            hits,
            plan,
            prefetch_k,
        }
    }

    pub fn search_graph_view_with_policy_metadata_rerank_timed(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        rerank: GraphRerankConfig,
    ) -> GraphTimedRerankedSearchReport {
        let total_start = Instant::now();
        let view_start = Instant::now();
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let view_build_ns = view_start.elapsed().as_nanos();
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);

        let mut vector_search_ns = 0;
        let mut rerank_ns = 0;
        let mut blocks_skipped_by_mask = 0;
        let mut hits = Vec::new();

        if prefetch_k > 0 {
            assert_eq!(
                query.len(),
                self.dim(),
                "GraphMemoryIndex search expects a single query vector"
            );
            let skipped_before = crate::search::blocks_skipped_by_mask();
            let search_start = Instant::now();
            let results = self.index.search_with_slot_mask(query, prefetch_k, &mask);
            vector_search_ns = search_start.elapsed().as_nanos();
            let skipped_after = crate::search::blocks_skipped_by_mask();
            blocks_skipped_by_mask = skipped_after.saturating_sub(skipped_before);

            let rerank_start = Instant::now();
            hits = self.reranked_hits_from_results(&results, &path_by_slot, rerank);
            hits.truncate(k);
            rerank_ns = rerank_start.elapsed().as_nanos();
        }

        let total_ns = total_start.elapsed().as_nanos();
        GraphTimedRerankedSearchReport {
            hits,
            plan,
            prefetch_k,
            telemetry: GraphSearchTelemetry {
                view_build_ns,
                vector_search_ns,
                rerank_ns,
                trace_build_ns: 0,
                total_ns,
                blocks_skipped_by_mask,
            },
        }
    }

    pub fn search_graph_view_with_policy_metadata_candidates_rerank_timed(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
        rerank: GraphRerankConfig,
    ) -> GraphCandidateTimedRerankedSearchReport {
        let total_start = Instant::now();
        let view_start = Instant::now();
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_candidates_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_ids,
        );
        let view_build_ns = view_start.elapsed().as_nanos();
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);

        let mut vector_search_ns = 0;
        let mut rerank_ns = 0;
        let mut blocks_skipped_by_mask = 0;
        let mut hits = Vec::new();

        if prefetch_k > 0 {
            assert_eq!(
                query.len(),
                self.dim(),
                "GraphMemoryIndex search expects a single query vector"
            );
            let skipped_before = crate::search::blocks_skipped_by_mask();
            let search_start = Instant::now();
            let results = self.index.search_with_slot_mask(query, prefetch_k, &mask);
            vector_search_ns = search_start.elapsed().as_nanos();
            let skipped_after = crate::search::blocks_skipped_by_mask();
            blocks_skipped_by_mask = skipped_after.saturating_sub(skipped_before);

            let rerank_start = Instant::now();
            hits = self.reranked_hits_from_results(&results, &path_by_slot, rerank);
            hits.truncate(k);
            rerank_ns = rerank_start.elapsed().as_nanos();
        }

        let total_ns = total_start.elapsed().as_nanos();
        GraphCandidateTimedRerankedSearchReport {
            hits,
            plan,
            prefetch_k,
            telemetry: GraphSearchTelemetry {
                view_build_ns,
                vector_search_ns,
                rerank_ns,
                trace_build_ns: 0,
                total_ns,
                blocks_skipped_by_mask,
            },
        }
    }

    pub fn search_graph_view_with_policy_metadata_candidate_scores_hybrid_timed(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_scores: &[(u64, f32)],
        rerank: GraphHybridRerankConfig,
    ) -> GraphCandidateTimedHybridSearchReport {
        let total_start = Instant::now();
        let view_start = Instant::now();
        let policy_key = PolicyGraphViewKey::new(seeds, policy);
        let visits = self.policy_visits(&policy_key);
        let path_by_slot = self.weighted_visits_by_slot(&visits);
        let (mask, plan, mut candidate_score_by_slot) = self
            .graph_view_mask_with_policy_metadata_candidate_scores_plan(
                seeds,
                policy,
                required_tags,
                allowed_sources,
                start_ms,
                end_ms,
                candidate_scores,
            );
        normalize_candidate_score_map(
            &mut candidate_score_by_slot,
            rerank.candidate_score_normalization,
            Some(&mask),
        );
        let view_build_ns = view_start.elapsed().as_nanos();
        let prefetch_k = rerank.prefetch_k(k, plan.selected_slots);

        let mut vector_search_ns = 0;
        let mut rerank_ns = 0;
        let mut blocks_skipped_by_mask = 0;
        let mut hits = Vec::new();

        if prefetch_k > 0 {
            assert_eq!(
                query.len(),
                self.dim(),
                "GraphMemoryIndex search expects a single query vector"
            );
            let skipped_before = crate::search::blocks_skipped_by_mask();
            let search_start = Instant::now();
            let results = self.index.search_with_slot_mask(query, prefetch_k, &mask);
            vector_search_ns = search_start.elapsed().as_nanos();
            let skipped_after = crate::search::blocks_skipped_by_mask();
            blocks_skipped_by_mask = skipped_after.saturating_sub(skipped_before);

            let rerank_start = Instant::now();
            hits = self.hybrid_hits_from_results(
                &results,
                &path_by_slot,
                &candidate_score_by_slot,
                rerank,
            );
            hits.truncate(k);
            rerank_ns = rerank_start.elapsed().as_nanos();
        }

        let total_ns = total_start.elapsed().as_nanos();
        GraphCandidateTimedHybridSearchReport {
            hits,
            plan,
            prefetch_k,
            telemetry: GraphSearchTelemetry {
                view_build_ns,
                vector_search_ns,
                rerank_ns,
                trace_build_ns: 0,
                total_ns,
                blocks_skipped_by_mask,
            },
        }
    }

    pub fn explain_graph_search_with_policy_metadata_rerank_timed(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        rerank: GraphRerankConfig,
    ) -> GraphExplainedSearchReport {
        let total_start = Instant::now();
        let timed = self.search_graph_view_with_policy_metadata_rerank_timed(
            query,
            k,
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            rerank,
        );
        let trace_start = Instant::now();
        let trace = self.explain_graph_view_with_policy(seeds, policy);
        let mut telemetry = timed.telemetry;
        telemetry.trace_build_ns = trace_start.elapsed().as_nanos();
        telemetry.total_ns = total_start.elapsed().as_nanos();

        GraphExplainedSearchReport {
            hits: timed.hits,
            plan: timed.plan,
            prefetch_k: timed.prefetch_k,
            telemetry,
            trace,
        }
    }

    pub fn explain_graph_search_with_policy_metadata_candidates_rerank_timed(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_ids: &[u64],
        rerank: GraphRerankConfig,
    ) -> GraphCandidateExplainedSearchReport {
        let total_start = Instant::now();
        let timed = self.search_graph_view_with_policy_metadata_candidates_rerank_timed(
            query,
            k,
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_ids,
            rerank,
        );
        let trace_start = Instant::now();
        let trace = self.explain_graph_view_with_policy(seeds, policy);
        let mut telemetry = timed.telemetry;
        telemetry.trace_build_ns = trace_start.elapsed().as_nanos();
        telemetry.total_ns = total_start.elapsed().as_nanos();

        GraphCandidateExplainedSearchReport {
            hits: timed.hits,
            plan: timed.plan,
            prefetch_k: timed.prefetch_k,
            telemetry,
            trace,
        }
    }

    pub fn explain_graph_search_with_policy_metadata_candidate_scores_hybrid_timed(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
        candidate_scores: &[(u64, f32)],
        rerank: GraphHybridRerankConfig,
    ) -> GraphCandidateHybridExplainedSearchReport {
        let total_start = Instant::now();
        let timed = self.search_graph_view_with_policy_metadata_candidate_scores_hybrid_timed(
            query,
            k,
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            candidate_scores,
            rerank,
        );
        let trace_start = Instant::now();
        let trace = self.explain_graph_view_with_policy(seeds, policy);
        let mut telemetry = timed.telemetry;
        telemetry.trace_build_ns = trace_start.elapsed().as_nanos();
        telemetry.total_ns = total_start.elapsed().as_nanos();

        GraphCandidateHybridExplainedSearchReport {
            hits: timed.hits,
            plan: timed.plan,
            prefetch_k: timed.prefetch_k,
            telemetry,
            trace,
        }
    }

    pub fn explain_graph_search_with_preset(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        preset: GraphSearchPreset,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphExplainedSearchReport {
        let tuning = self.tuning_for_preset(k, preset);
        self.explain_graph_search_with_policy_metadata_rerank_timed(
            query,
            k,
            seeds,
            tuning.policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
            tuning.rerank,
        )
    }

    pub fn search_graph_view_with_trace(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
    ) -> GraphSearchTraceReport {
        let trace = self.explain_graph_view(seeds, max_hops);
        let mut mask = SlotMask::from_slots(self.len(), trace.nodes.iter().map(|node| node.slot));
        for tag in required_tags {
            let tag_mask = self.tag_view_mask(tag);
            mask.intersect_with(&tag_mask);
        }
        let mut view = trace.stats;
        view.selected_slots = mask.count();
        let hits = self.search_with_slot_mask(query, k, &mask);
        GraphSearchTraceReport { hits, view, trace }
    }

    pub fn search_graph_view_with_metadata_trace(
        &mut self,
        query: &[f32],
        k: usize,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphSearchTraceReport {
        let trace = self.explain_graph_view(seeds, max_hops);
        let mut mask = SlotMask::from_slots(self.len(), trace.nodes.iter().map(|node| node.slot));
        self.apply_metadata_filters(&mut mask, required_tags, allowed_sources, start_ms, end_ms);
        let mut view = trace.stats;
        view.selected_slots = mask.count();
        let hits = self.search_with_slot_mask(query, k, &mask);
        GraphSearchTraceReport { hits, view, trace }
    }

    pub fn prepare_graph_view(&mut self, seeds: &[u64], max_hops: usize) -> GraphPreparedView {
        let (mask, view) = self.graph_view_mask_with_stats(seeds, max_hops);
        let plan = GraphSearchPlan {
            total_slots: self.len(),
            graph_slots: view.selected_slots,
            selected_slots: view.selected_slots,
            active_blocks: mask.active_block_count(),
            graph_cache_hit: view.cache_hit,
            combined_cache_hit: false,
        };
        GraphPreparedView { plan, mask }
    }

    pub fn prepare_graph_view_with_metadata(
        &mut self,
        seeds: &[u64],
        max_hops: usize,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphPreparedView {
        let (mask, plan) = self.graph_view_mask_with_metadata_plan(
            seeds,
            max_hops,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        GraphPreparedView { plan, mask }
    }

    pub fn prepare_graph_view_with_policy(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
    ) -> GraphPreparedPolicyView {
        let key = PolicyGraphViewKey::new(seeds, policy);
        let (mask, view) = self.graph_view_mask_with_policy_stats(&key.seeds, policy);
        let plan = GraphSearchPlan {
            total_slots: self.len(),
            graph_slots: view.selected_slots,
            selected_slots: view.selected_slots,
            active_blocks: mask.active_block_count(),
            graph_cache_hit: view.cache_hit,
            combined_cache_hit: false,
        };
        let path = self.policy_visits(&key);
        let path_by_slot = self.weighted_visits_by_slot(&path);
        GraphPreparedPolicyView {
            plan,
            mask,
            path_by_slot,
        }
    }

    pub fn prepare_graph_view_with_policy_metadata(
        &mut self,
        seeds: &[u64],
        policy: GraphViewPolicy,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) -> GraphPreparedPolicyView {
        let key = PolicyGraphViewKey::new(seeds, policy);
        let (mask, plan) = self.graph_view_mask_with_policy_metadata_plan(
            seeds,
            policy,
            required_tags,
            allowed_sources,
            start_ms,
            end_ms,
        );
        let path = self.policy_visits(&key);
        let path_by_slot = self.weighted_visits_by_slot(&path);
        GraphPreparedPolicyView {
            plan,
            mask,
            path_by_slot,
        }
    }

    pub fn search_with_slot_mask(
        &self,
        query: &[f32],
        k: usize,
        mask: &SlotMask,
    ) -> Vec<MemoryHit> {
        if k == 0 {
            return Vec::new();
        }
        if mask.is_empty() {
            return Vec::new();
        }
        assert_eq!(
            query.len(),
            self.dim(),
            "GraphMemoryIndex search expects a single query vector"
        );
        let results = self.index.search_with_slot_mask(query, k, mask);
        self.hits_from_results(&results)
    }

    pub fn search_with_slot_mask_batch(
        &self,
        queries: &[f32],
        k: usize,
        mask: &SlotMask,
    ) -> Vec<Vec<MemoryHit>> {
        if k == 0 {
            return vec![Vec::new(); query_count_for_dim(queries, self.dim())];
        }
        let results = self.index.search_with_slot_mask(queries, k, mask);
        self.hits_by_query_from_results(&results)
    }

    pub fn search_global(&self, query: &[f32], k: usize) -> Vec<MemoryHit> {
        assert_eq!(
            query.len(),
            self.dim(),
            "GraphMemoryIndex search expects a single query vector"
        );
        let results = self.index.search(query, k);
        self.hits_from_results(&results)
    }

    pub fn write(
        &self,
        index_path: impl AsRef<Path>,
        graph_path: impl AsRef<Path>,
    ) -> Result<(), GraphMemoryError> {
        self.index.write(index_path)?;
        self.write_graph_sidecar(graph_path)?;
        Ok(())
    }

    pub fn load(
        index_path: impl AsRef<Path>,
        graph_path: impl AsRef<Path>,
    ) -> Result<Self, GraphMemoryError> {
        let index = TurboQuantIndex::load(index_path)?;
        let (sidecar_dim, sidecar_bit_width, slot_to_id, records, edges) =
            read_graph_sidecar(graph_path)?;
        if sidecar_dim != index.dim() || sidecar_bit_width != index.bit_width() {
            return Err(GraphMemoryError::CorruptSidecar(format!(
                "sidecar dim/bit_width ({sidecar_dim}/{sidecar_bit_width}) does not match index ({}/{})",
                index.dim(),
                index.bit_width(),
            )));
        }
        if slot_to_id.len() != index.len() {
            return Err(GraphMemoryError::CorruptSidecar(format!(
                "sidecar has {} slots but index has {} vectors",
                slot_to_id.len(),
                index.len()
            )));
        }
        let mut id_to_slot = HashMap::with_capacity(slot_to_id.len());
        for (slot, &id) in slot_to_id.iter().enumerate() {
            if id_to_slot.insert(id, slot).is_some() {
                return Err(GraphMemoryError::CorruptSidecar(format!(
                    "duplicate id {id} in slot table"
                )));
            }
            if !records.contains_key(&id) {
                return Err(GraphMemoryError::CorruptSidecar(format!(
                    "slot id {id} is missing its record"
                )));
            }
        }
        for id in records.keys() {
            if !id_to_slot.contains_key(id) {
                return Err(GraphMemoryError::CorruptSidecar(format!(
                    "record id {id} is not present in slot table"
                )));
            }
        }
        for (&from, neighbors) in &edges {
            if !id_to_slot.contains_key(&from) {
                return Err(GraphMemoryError::CorruptSidecar(format!(
                    "edge source id {from} is not present"
                )));
            }
            for edge in neighbors {
                if !id_to_slot.contains_key(&edge.to) {
                    return Err(GraphMemoryError::CorruptSidecar(format!(
                        "edge target id {} is not present",
                        edge.to
                    )));
                }
            }
        }

        Ok(Self {
            index,
            slot_to_id,
            id_to_slot,
            tag_to_ids: build_tag_index(&records),
            source_to_ids: build_source_index(&records),
            time_index: build_time_index(&records),
            records,
            edges,
            tag_cache: HashMap::new(),
            source_cache: HashMap::new(),
            time_cache: HashMap::new(),
            view_cache: HashMap::new(),
            policy_visit_cache: HashMap::new(),
            policy_view_cache: HashMap::new(),
            combined_view_cache: HashMap::new(),
            combined_policy_view_cache: HashMap::new(),
            cache_access: GraphMemoryCacheAccess::default(),
        })
    }

    pub fn len(&self) -> usize {
        self.slot_to_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slot_to_id.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.index.dim()
    }

    pub fn bit_width(&self) -> usize {
        self.index.bit_width()
    }

    pub fn contains(&self, id: u64) -> bool {
        self.id_to_slot.contains_key(&id)
    }

    pub fn slot_of(&self, id: u64) -> Option<usize> {
        self.id_to_slot.get(&id).copied()
    }

    pub fn record(&self, id: u64) -> Option<&MemoryRecord> {
        self.records.get(&id)
    }

    pub fn neighbors(&self, id: u64) -> &[MemoryEdge] {
        self.edges.get(&id).map_or(&[], Vec::as_slice)
    }

    pub fn prepare(&self) {
        self.index.prepare();
    }

    pub fn cache_stats(&self) -> GraphMemoryCacheStats {
        let mut stats = GraphMemoryCacheStats {
            graph_views: self.view_cache.len(),
            policy_visits: self.policy_visit_cache.len(),
            policy_views: self.policy_view_cache.len(),
            combined_views: self.combined_view_cache.len(),
            combined_policy_views: self.combined_policy_view_cache.len(),
            tag_masks: self.tag_cache.len(),
            source_masks: self.source_cache.len(),
            time_masks: self.time_cache.len(),
            graph_view_hits: self.cache_access.graph_views.hits,
            graph_view_misses: self.cache_access.graph_views.misses,
            policy_visit_hits: self.cache_access.policy_visits.hits,
            policy_visit_misses: self.cache_access.policy_visits.misses,
            policy_view_hits: self.cache_access.policy_views.hits,
            policy_view_misses: self.cache_access.policy_views.misses,
            combined_view_hits: self.cache_access.combined_views.hits,
            combined_view_misses: self.cache_access.combined_views.misses,
            combined_policy_view_hits: self.cache_access.combined_policy_views.hits,
            combined_policy_view_misses: self.cache_access.combined_policy_views.misses,
            tag_mask_hits: self.cache_access.tag_masks.hits,
            tag_mask_misses: self.cache_access.tag_masks.misses,
            source_mask_hits: self.cache_access.source_masks.hits,
            source_mask_misses: self.cache_access.source_masks.misses,
            time_mask_hits: self.cache_access.time_masks.hits,
            time_mask_misses: self.cache_access.time_masks.misses,
            total_entries: 0,
        };
        stats.total_entries = stats.query_entries() + stats.metadata_entries();
        stats
    }

    pub fn cache_budget_for_preset(&self, preset: GraphSearchPreset) -> GraphMemoryCacheBudget {
        preset.cache_budget(self.len())
    }

    /// Drop graph/search-result view caches while keeping metadata masks.
    pub fn clear_query_caches(&mut self) {
        self.view_cache.clear();
        self.policy_visit_cache.clear();
        self.policy_view_cache.clear();
        self.combined_view_cache.clear();
        self.combined_policy_view_cache.clear();
    }

    /// Drop tag/source/time masks while keeping graph-view caches.
    pub fn clear_metadata_caches(&mut self) {
        self.tag_cache.clear();
        self.source_cache.clear();
        self.time_cache.clear();
    }

    pub fn clear_all_caches(&mut self) {
        self.clear_query_caches();
        self.clear_metadata_caches();
    }

    /// Bound each graph/search-result cache to at most `max_entries_per_cache`.
    ///
    /// This is a simple memory cap, not an LRU policy; it keeps an arbitrary
    /// subset of each internal cache.
    pub fn trim_query_caches(&mut self, max_entries_per_cache: usize) {
        trim_hash_map(&mut self.view_cache, max_entries_per_cache);
        trim_hash_map(&mut self.policy_visit_cache, max_entries_per_cache);
        trim_hash_map(&mut self.policy_view_cache, max_entries_per_cache);
        trim_hash_map(&mut self.combined_view_cache, max_entries_per_cache);
        trim_hash_map(&mut self.combined_policy_view_cache, max_entries_per_cache);
    }

    /// Bound each metadata mask cache to at most `max_entries_per_cache`.
    pub fn trim_metadata_caches(&mut self, max_entries_per_cache: usize) {
        trim_hash_map(&mut self.tag_cache, max_entries_per_cache);
        trim_hash_map(&mut self.source_cache, max_entries_per_cache);
        trim_hash_map(&mut self.time_cache, max_entries_per_cache);
    }

    pub fn trim_all_caches(&mut self, max_entries_per_cache: usize) {
        self.trim_query_caches(max_entries_per_cache);
        self.trim_metadata_caches(max_entries_per_cache);
    }

    pub fn trim_caches_to_budget(&mut self, budget: GraphMemoryCacheBudget) {
        trim_hash_map(&mut self.view_cache, budget.graph_views);
        trim_hash_map(&mut self.policy_visit_cache, budget.policy_visits);
        trim_hash_map(&mut self.policy_view_cache, budget.policy_views);
        trim_hash_map(&mut self.combined_view_cache, budget.combined_views);
        trim_hash_map(
            &mut self.combined_policy_view_cache,
            budget.combined_policy_views,
        );
        trim_hash_map(&mut self.tag_cache, budget.tag_masks);
        trim_hash_map(&mut self.source_cache, budget.source_masks);
        trim_hash_map(&mut self.time_cache, budget.time_masks);
    }

    pub fn trim_caches_for_preset(&mut self, preset: GraphSearchPreset) -> GraphMemoryCacheBudget {
        let budget = self.cache_budget_for_preset(preset);
        self.trim_caches_to_budget(budget);
        budget
    }

    fn hits_from_results(&self, results: &crate::SearchResults) -> Vec<MemoryHit> {
        self.hits_from_results_for_query(results, 0)
    }

    fn hits_by_query_from_results(&self, results: &crate::SearchResults) -> Vec<Vec<MemoryHit>> {
        (0..results.nq)
            .map(|qi| self.hits_from_results_for_query(results, qi))
            .collect()
    }

    fn hits_from_results_for_query(
        &self,
        results: &crate::SearchResults,
        qi: usize,
    ) -> Vec<MemoryHit> {
        results
            .indices_for_query(qi)
            .iter()
            .zip(results.scores_for_query(qi))
            .map(|(&slot, &score)| {
                let id = self.slot_to_id[slot as usize];
                let record = self
                    .records
                    .get(&id)
                    .expect("slot table and records are in sync");
                MemoryHit {
                    id,
                    score,
                    title: record.title.clone(),
                    tags: record.tags.clone(),
                    source: record.source.clone(),
                    timestamp_ms: record.timestamp_ms,
                }
            })
            .collect()
    }

    fn reranked_hits_from_results(
        &self,
        results: &crate::SearchResults,
        path_by_slot: &HashMap<usize, WeightedVisit>,
        rerank: GraphRerankConfig,
    ) -> Vec<GraphRerankedHit> {
        self.reranked_hits_from_results_for_query(results, 0, path_by_slot, rerank)
    }

    fn reranked_hits_from_results_for_query(
        &self,
        results: &crate::SearchResults,
        qi: usize,
        path_by_slot: &HashMap<usize, WeightedVisit>,
        rerank: GraphRerankConfig,
    ) -> Vec<GraphRerankedHit> {
        let rerank = rerank.normalized();
        let mut hits: Vec<GraphRerankedHit> = results
            .indices_for_query(qi)
            .iter()
            .zip(results.scores_for_query(qi))
            .filter_map(|(&slot, &vector_score)| {
                let slot = slot as usize;
                let visit = path_by_slot.get(&slot)?;
                let id = self.slot_to_id[slot];
                let record = self
                    .records
                    .get(&id)
                    .expect("slot table and records are in sync");
                let graph_score = visit.path_weight;
                let score = rerank.vector_weight * vector_score + rerank.graph_weight * graph_score;
                Some(GraphRerankedHit {
                    id,
                    score,
                    vector_score,
                    graph_score,
                    depth: visit.depth,
                    parent: visit.parent,
                    title: record.title.clone(),
                    tags: record.tags.clone(),
                    source: record.source.clone(),
                    timestamp_ms: record.timestamp_ms,
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.vector_score.total_cmp(&a.vector_score))
                .then_with(|| b.graph_score.total_cmp(&a.graph_score))
                .then_with(|| a.id.cmp(&b.id))
        });
        hits
    }

    fn hybrid_hits_from_results(
        &self,
        results: &crate::SearchResults,
        path_by_slot: &HashMap<usize, WeightedVisit>,
        candidate_score_by_slot: &HashMap<usize, f32>,
        rerank: GraphHybridRerankConfig,
    ) -> Vec<GraphHybridHit> {
        self.hybrid_hits_from_results_for_query(
            results,
            0,
            path_by_slot,
            candidate_score_by_slot,
            rerank,
        )
    }

    fn hybrid_hits_from_results_for_query(
        &self,
        results: &crate::SearchResults,
        qi: usize,
        path_by_slot: &HashMap<usize, WeightedVisit>,
        candidate_score_by_slot: &HashMap<usize, f32>,
        rerank: GraphHybridRerankConfig,
    ) -> Vec<GraphHybridHit> {
        let rerank = rerank.normalized();
        let mut hits: Vec<GraphHybridHit> = results
            .indices_for_query(qi)
            .iter()
            .zip(results.scores_for_query(qi))
            .filter_map(|(&slot, &vector_score)| {
                let slot = slot as usize;
                let visit = path_by_slot.get(&slot)?;
                let id = self.slot_to_id[slot];
                let record = self
                    .records
                    .get(&id)
                    .expect("slot table and records are in sync");
                let graph_score = visit.path_weight;
                let candidate_score = candidate_score_by_slot.get(&slot).copied().unwrap_or(0.0);
                let score = rerank.vector_weight * vector_score
                    + rerank.graph_weight * graph_score
                    + rerank.candidate_weight * candidate_score;
                Some(GraphHybridHit {
                    id,
                    score,
                    vector_score,
                    graph_score,
                    candidate_score,
                    depth: visit.depth,
                    parent: visit.parent,
                    title: record.title.clone(),
                    tags: record.tags.clone(),
                    source: record.source.clone(),
                    timestamp_ms: record.timestamp_ms,
                })
            })
            .collect();

        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| b.vector_score.total_cmp(&a.vector_score))
                .then_with(|| b.graph_score.total_cmp(&a.graph_score))
                .then_with(|| b.candidate_score.total_cmp(&a.candidate_score))
                .then_with(|| a.id.cmp(&b.id))
        });
        hits
    }

    fn build_graph_view_trace(&self, key: GraphViewKey, stats: GraphViewStats) -> GraphViewTrace {
        let mut nodes = Vec::new();
        let mut selected = HashSet::new();
        let mut queue = VecDeque::new();

        for &seed in &key.seeds {
            if self.id_to_slot.contains_key(&seed) && selected.insert(seed) {
                queue.push_back((seed, 0usize, None, 1.0f32, 1.0f32));
            }
        }

        while let Some((id, depth, parent, via_weight, path_weight)) = queue.pop_front() {
            if let (Some(&slot), Some(record)) = (self.id_to_slot.get(&id), self.records.get(&id)) {
                nodes.push(GraphViewNode {
                    id,
                    slot,
                    depth,
                    parent,
                    via_weight,
                    path_weight,
                    title: record.title.clone(),
                    tags: record.tags.clone(),
                    source: record.source.clone(),
                    timestamp_ms: record.timestamp_ms,
                });
            }
            if depth == key.max_hops {
                continue;
            }
            if let Some(neighbors) = self.edges.get(&id) {
                for edge in neighbors {
                    if selected.insert(edge.to) {
                        queue.push_back((
                            edge.to,
                            depth + 1,
                            Some(id),
                            edge.weight,
                            path_weight * edge.weight,
                        ));
                    }
                }
            }
        }

        let mut edges = Vec::new();
        for (&from, neighbors) in &self.edges {
            if !selected.contains(&from) {
                continue;
            }
            for edge in neighbors {
                if selected.contains(&edge.to) {
                    edges.push(GraphViewEdge {
                        from,
                        to: edge.to,
                        weight: edge.weight,
                    });
                }
            }
        }

        GraphViewTrace {
            seeds: key.seeds,
            max_hops: key.max_hops,
            stats,
            nodes,
            edges,
        }
    }

    fn build_policy_graph_view_trace(
        &self,
        key: PolicyGraphViewKey,
        stats: GraphViewStats,
        visits: &[WeightedVisit],
    ) -> GraphViewTrace {
        let selected: HashSet<u64> = visits.iter().map(|visit| visit.id).collect();
        let mut nodes = Vec::with_capacity(visits.len());

        for visit in visits {
            if let (Some(&slot), Some(record)) =
                (self.id_to_slot.get(&visit.id), self.records.get(&visit.id))
            {
                nodes.push(GraphViewNode {
                    id: visit.id,
                    slot,
                    depth: visit.depth,
                    parent: visit.parent,
                    via_weight: visit.via_weight,
                    path_weight: visit.path_weight,
                    title: record.title.clone(),
                    tags: record.tags.clone(),
                    source: record.source.clone(),
                    timestamp_ms: record.timestamp_ms,
                });
            }
        }

        let mut edges = Vec::new();
        for (&from, neighbors) in &self.edges {
            if !selected.contains(&from) {
                continue;
            }
            for edge in neighbors {
                if selected.contains(&edge.to) {
                    edges.push(GraphViewEdge {
                        from,
                        to: edge.to,
                        weight: edge.weight,
                    });
                }
            }
        }

        GraphViewTrace {
            seeds: key.seeds,
            max_hops: key.max_hops,
            stats,
            nodes,
            edges,
        }
    }

    fn policy_visits(&mut self, key: &PolicyGraphViewKey) -> Vec<WeightedVisit> {
        if let Some(visits) = self.policy_visit_cache.get(key) {
            self.cache_access.policy_visits.record_hit();
            return visits.clone();
        }
        self.cache_access.policy_visits.record_miss();
        let visits = self.weighted_graph_visits(key);
        self.policy_visit_cache.insert(key.clone(), visits.clone());
        visits
    }

    fn weighted_graph_visits(&self, key: &PolicyGraphViewKey) -> Vec<WeightedVisit> {
        if key.max_nodes == 0 || key.max_active_blocks == 0 {
            return Vec::new();
        }

        let min_path_weight = key.min_path_weight();
        let mut heap = BinaryHeap::new();
        let mut selected = HashSet::new();
        let mut active_blocks = HashSet::new();
        let mut visits = Vec::new();

        for &seed in &key.seeds {
            if self.id_to_slot.contains_key(&seed) {
                heap.push(WeightedQueueItem {
                    id: seed,
                    depth: 0,
                    parent: None,
                    via_weight: 1.0,
                    path_weight: 1.0,
                });
            }
        }

        while let Some(item) = heap.pop() {
            if visits.len() >= key.max_nodes {
                break;
            }
            let Some(&slot) = self.id_to_slot.get(&item.id) else {
                continue;
            };
            if selected.contains(&item.id) {
                continue;
            }
            let block = slot / BLOCK;
            if !active_blocks.contains(&block) && active_blocks.len() >= key.max_active_blocks {
                continue;
            }
            selected.insert(item.id);
            active_blocks.insert(block);

            visits.push(WeightedVisit {
                id: item.id,
                depth: item.depth,
                parent: item.parent,
                via_weight: item.via_weight,
                path_weight: item.path_weight,
            });

            if item.depth == key.max_hops {
                continue;
            }
            if let Some(neighbors) = self.edges.get(&item.id) {
                for edge in neighbors {
                    if selected.contains(&edge.to) {
                        continue;
                    }
                    let Some(edge_weight) = normalize_edge_weight(edge.weight) else {
                        continue;
                    };
                    let path_weight = (item.path_weight * edge_weight).min(1.0);
                    if path_weight < min_path_weight {
                        continue;
                    }
                    heap.push(WeightedQueueItem {
                        id: edge.to,
                        depth: item.depth + 1,
                        parent: Some(item.id),
                        via_weight: edge_weight,
                        path_weight,
                    });
                }
            }
        }

        visits
    }

    fn weighted_visits_by_slot(&self, visits: &[WeightedVisit]) -> HashMap<usize, WeightedVisit> {
        let mut by_slot = HashMap::with_capacity(visits.len());
        for &visit in visits {
            if let Some(&slot) = self.id_to_slot.get(&visit.id) {
                by_slot.insert(slot, visit);
            }
        }
        by_slot
    }

    fn candidate_score_mask_with_stats(
        &self,
        candidate_scores: &[(u64, f32)],
    ) -> CandidateScoreMaskBuild {
        let mut mask = SlotMask::new(self.len());
        let mut seen = HashSet::with_capacity(candidate_scores.len());
        let mut missing_ids = 0;
        let mut duplicate_ids = 0;
        let mut score_by_slot = HashMap::with_capacity(candidate_scores.len());

        for &(id, score) in candidate_scores {
            let first_seen = seen.insert(id);
            if !first_seen {
                duplicate_ids += 1;
            }

            let Some(&slot) = self.id_to_slot.get(&id) else {
                if first_seen {
                    missing_ids += 1;
                }
                continue;
            };
            if first_seen {
                mask.allow(slot);
            }

            let score = finite_or(score, 0.0);
            score_by_slot
                .entry(slot)
                .and_modify(|stored| {
                    if score > *stored {
                        *stored = score;
                    }
                })
                .or_insert(score);
        }

        CandidateScoreMaskBuild {
            candidate: CandidateMaskBuild {
                input_ids: candidate_scores.len(),
                candidate_slots: mask.count(),
                missing_ids,
                duplicate_ids,
                mask,
            },
            score_by_slot,
        }
    }

    fn candidate_id_mask_with_stats(&self, ids: &[u64]) -> CandidateMaskBuild {
        let mut mask = SlotMask::new(self.len());
        let mut seen = HashSet::with_capacity(ids.len());
        let mut missing_ids = 0;
        let mut duplicate_ids = 0;

        for &id in ids {
            if !seen.insert(id) {
                duplicate_ids += 1;
                continue;
            }
            let Some(&slot) = self.id_to_slot.get(&id) else {
                missing_ids += 1;
                continue;
            };
            mask.allow(slot);
        }

        CandidateMaskBuild {
            input_ids: ids.len(),
            candidate_slots: mask.count(),
            missing_ids,
            duplicate_ids,
            mask,
        }
    }

    fn require_id(&self, id: u64) -> Result<(), GraphMemoryError> {
        if self.id_to_slot.contains_key(&id) {
            Ok(())
        } else {
            Err(GraphMemoryError::MissingId(id))
        }
    }

    fn apply_metadata_filters(
        &mut self,
        mask: &mut SlotMask,
        required_tags: &[&str],
        allowed_sources: &[&str],
        start_ms: Option<i64>,
        end_ms: Option<i64>,
    ) {
        let mut intersections = Vec::new();
        for tag in required_tags {
            intersections.push(self.tag_view_mask(tag));
        }

        if !allowed_sources.is_empty() {
            let mut source_union = SlotMask::new(self.len());
            let mut source_masks = Vec::with_capacity(allowed_sources.len());
            for source in allowed_sources {
                source_masks.push(self.source_view_mask(source));
            }
            source_union.union_with_many(source_masks.iter());
            intersections.push(source_union);
        }

        if start_ms.is_some() || end_ms.is_some() {
            intersections.push(self.time_range_view_mask(start_ms, end_ms));
        }

        if !intersections.is_empty() {
            mask.intersect_with_many(intersections.iter());
        }
    }

    fn index_record_metadata(&mut self, record: &MemoryRecord) {
        for tag in &record.tags {
            self.tag_to_ids
                .entry(tag.clone())
                .or_default()
                .push(record.id);
        }
        if let Some(source) = &record.source {
            self.source_to_ids
                .entry(source.clone())
                .or_default()
                .push(record.id);
        }
        if let Some(timestamp_ms) = record.timestamp_ms {
            let entry = (timestamp_ms, record.id);
            let pos = self
                .time_index
                .binary_search(&entry)
                .unwrap_or_else(|pos| pos);
            self.time_index.insert(pos, entry);
        }
    }

    fn unindex_record_metadata(&mut self, record: &MemoryRecord) {
        for tag in &record.tags {
            let should_remove = match self.tag_to_ids.get_mut(tag) {
                Some(ids) => {
                    ids.retain(|&candidate| candidate != record.id);
                    ids.is_empty()
                }
                None => false,
            };
            if should_remove {
                self.tag_to_ids.remove(tag);
            }
        }
        if let Some(source) = &record.source {
            let should_remove = match self.source_to_ids.get_mut(source) {
                Some(ids) => {
                    ids.retain(|&candidate| candidate != record.id);
                    ids.is_empty()
                }
                None => false,
            };
            if should_remove {
                self.source_to_ids.remove(source);
            }
        }
        if let Some(timestamp_ms) = record.timestamp_ms {
            if let Ok(pos) = self.time_index.binary_search(&(timestamp_ms, record.id)) {
                self.time_index.remove(pos);
            }
        }
    }

    fn invalidate_views(&mut self) {
        self.view_cache.clear();
        self.tag_cache.clear();
        self.source_cache.clear();
        self.time_cache.clear();
        self.combined_view_cache.clear();
        self.policy_visit_cache.clear();
        self.policy_view_cache.clear();
        self.combined_policy_view_cache.clear();
    }

    fn invalidate_metadata_views(&mut self) {
        self.tag_cache.clear();
        self.source_cache.clear();
        self.time_cache.clear();
        self.combined_view_cache.clear();
        self.combined_policy_view_cache.clear();
    }

    fn write_graph_sidecar(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);
        f.write_all(GRAPH_MAGIC)?;
        f.write_all(&[GRAPH_VERSION])?;
        write_u32(&mut f, self.dim() as u32)?;
        f.write_all(&[self.bit_width() as u8])?;
        write_u64(&mut f, self.slot_to_id.len() as u64)?;
        for &id in &self.slot_to_id {
            write_u64(&mut f, id)?;
            let record = self
                .records
                .get(&id)
                .expect("slot table and records are in sync");
            write_string(&mut f, &record.title)?;
            write_u32(&mut f, record.tags.len() as u32)?;
            for tag in &record.tags {
                write_string(&mut f, tag)?;
            }
            match &record.source {
                Some(source) => {
                    f.write_all(&[1])?;
                    write_string(&mut f, source)?;
                }
                None => f.write_all(&[0])?,
            }
            match record.timestamp_ms {
                Some(timestamp_ms) => {
                    f.write_all(&[1])?;
                    write_i64(&mut f, timestamp_ms)?;
                }
                None => f.write_all(&[0])?,
            }
        }

        let edge_count: usize = self.edges.values().map(Vec::len).sum();
        write_u64(&mut f, edge_count as u64)?;
        for (&from, neighbors) in &self.edges {
            for edge in neighbors {
                write_u64(&mut f, from)?;
                write_u64(&mut f, edge.to)?;
                f.write_all(&edge.weight.to_le_bytes())?;
            }
        }
        f.flush()
    }
}

type SidecarLoad = (
    usize,
    usize,
    Vec<u64>,
    HashMap<u64, MemoryRecord>,
    HashMap<u64, Vec<MemoryEdge>>,
);

#[derive(Clone, Copy, Debug)]
struct WeightedVisit {
    id: u64,
    depth: usize,
    parent: Option<u64>,
    via_weight: f32,
    path_weight: f32,
}

#[derive(Clone, Copy, Debug)]
struct WeightedQueueItem {
    id: u64,
    depth: usize,
    parent: Option<u64>,
    via_weight: f32,
    path_weight: f32,
}

impl PartialEq for WeightedQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.depth == other.depth
            && self.parent == other.parent
            && self.path_weight.to_bits() == other.path_weight.to_bits()
    }
}

impl Eq for WeightedQueueItem {}

impl PartialOrd for WeightedQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WeightedQueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.path_weight
            .total_cmp(&other.path_weight)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.id.cmp(&self.id))
    }
}

fn normalize_min_path_weight(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn normalize_edge_weight(value: f32) -> Option<f32> {
    if value.is_finite() && value > 0.0 {
        Some(value.min(1.0))
    } else {
        None
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

fn normalize_candidate_score_map(
    scores: &mut HashMap<usize, f32>,
    normalization: GraphCandidateScoreNormalization,
    allowed_slots: Option<&SlotMask>,
) {
    if scores.is_empty() {
        return;
    }

    match normalization {
        GraphCandidateScoreNormalization::None => {}
        GraphCandidateScoreNormalization::MinMax => {
            let mut min_score = f32::INFINITY;
            let mut max_score = f32::NEG_INFINITY;
            for (&slot, &score) in scores.iter() {
                if !candidate_score_slot_allowed(allowed_slots, slot) {
                    continue;
                }
                min_score = min_score.min(score);
                max_score = max_score.max(score);
            }
            if !min_score.is_finite() {
                return;
            }

            if max_score > min_score {
                let span = max_score - min_score;
                for (&slot, score) in scores.iter_mut() {
                    if !candidate_score_slot_allowed(allowed_slots, slot) {
                        continue;
                    }
                    *score = (*score - min_score) / span;
                }
            } else {
                for (&slot, score) in scores.iter_mut() {
                    if !candidate_score_slot_allowed(allowed_slots, slot) {
                        continue;
                    }
                    *score = 1.0;
                }
            }
        }
        GraphCandidateScoreNormalization::MaxAbs => {
            let max_abs = scores.iter().fold(0.0f32, |max_abs, (&slot, score)| {
                if candidate_score_slot_allowed(allowed_slots, slot) {
                    max_abs.max(score.abs())
                } else {
                    max_abs
                }
            });
            if max_abs > 0.0 {
                for (&slot, score) in scores.iter_mut() {
                    if !candidate_score_slot_allowed(allowed_slots, slot) {
                        continue;
                    }
                    *score /= max_abs;
                }
            }
        }
    }
}

fn candidate_score_slot_allowed(allowed_slots: Option<&SlotMask>, slot: usize) -> bool {
    allowed_slots.map_or(true, |mask| slot < mask.len() && mask.contains(slot))
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn query_count_for_dim(queries: &[f32], dim: usize) -> usize {
    let nq = queries.len() / dim;
    assert_eq!(
        queries.len(),
        nq * dim,
        "query length {} is not a multiple of dim {}",
        queries.len(),
        dim,
    );
    nq
}

fn trim_hash_map<K, V>(map: &mut HashMap<K, V>, max_entries: usize)
where
    K: Eq + Hash + Clone,
{
    if map.len() <= max_entries {
        return;
    }
    let remove_count = map.len() - max_entries;
    let keys: Vec<K> = map.keys().take(remove_count).cloned().collect();
    for key in keys {
        map.remove(&key);
    }
}

fn normalized_strings(values: &[&str]) -> Vec<String> {
    let mut values: Vec<String> = values.iter().map(|value| (*value).to_string()).collect();
    values.sort_unstable();
    values.dedup();
    values
}

fn build_tag_index(records: &HashMap<u64, MemoryRecord>) -> HashMap<String, Vec<u64>> {
    let mut tag_to_ids: HashMap<String, Vec<u64>> = HashMap::new();
    for record in records.values() {
        for tag in &record.tags {
            tag_to_ids.entry(tag.clone()).or_default().push(record.id);
        }
    }
    for ids in tag_to_ids.values_mut() {
        ids.sort_unstable();
        ids.dedup();
    }
    tag_to_ids
}

fn build_source_index(records: &HashMap<u64, MemoryRecord>) -> HashMap<String, Vec<u64>> {
    let mut source_to_ids: HashMap<String, Vec<u64>> = HashMap::new();
    for record in records.values() {
        if let Some(source) = &record.source {
            source_to_ids
                .entry(source.clone())
                .or_default()
                .push(record.id);
        }
    }
    for ids in source_to_ids.values_mut() {
        ids.sort_unstable();
        ids.dedup();
    }
    source_to_ids
}

fn build_time_index(records: &HashMap<u64, MemoryRecord>) -> Vec<(i64, u64)> {
    let mut time_index = Vec::new();
    for record in records.values() {
        if let Some(timestamp_ms) = record.timestamp_ms {
            time_index.push((timestamp_ms, record.id));
        }
    }
    time_index.sort_unstable();
    time_index.dedup();
    time_index
}

fn read_graph_sidecar(path: impl AsRef<Path>) -> Result<SidecarLoad, GraphMemoryError> {
    let mut f = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != GRAPH_MAGIC {
        return Err(GraphMemoryError::CorruptSidecar(
            "wrong graph-memory magic".to_string(),
        ));
    }
    let mut version = [0u8; 1];
    f.read_exact(&mut version)?;
    if version[0] != 1 && version[0] != GRAPH_VERSION {
        return Err(GraphMemoryError::CorruptSidecar(format!(
            "unsupported graph-memory version {}",
            version[0]
        )));
    }
    let dim = read_u32(&mut f)? as usize;
    let mut bit_width = [0u8; 1];
    f.read_exact(&mut bit_width)?;
    let bit_width = bit_width[0] as usize;

    let n_nodes = read_u64(&mut f)? as usize;
    let mut slot_to_id = Vec::with_capacity(n_nodes);
    let mut records = HashMap::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        let id = read_u64(&mut f)?;
        let title = read_string(&mut f)?;
        let n_tags = read_u32(&mut f)? as usize;
        let mut tags = Vec::with_capacity(n_tags);
        for _ in 0..n_tags {
            tags.push(read_string(&mut f)?);
        }
        let (source, timestamp_ms) = if version[0] >= 2 {
            let source = if read_presence(&mut f)? {
                Some(read_string(&mut f)?)
            } else {
                None
            };
            let timestamp_ms = if read_presence(&mut f)? {
                Some(read_i64(&mut f)?)
            } else {
                None
            };
            (source, timestamp_ms)
        } else {
            (None, None)
        };
        slot_to_id.push(id);
        if records
            .insert(
                id,
                MemoryRecord {
                    id,
                    title,
                    tags,
                    source,
                    timestamp_ms,
                },
            )
            .is_some()
        {
            return Err(GraphMemoryError::CorruptSidecar(format!(
                "duplicate record id {id}"
            )));
        }
    }

    let n_edges = read_u64(&mut f)? as usize;
    let mut edges: HashMap<u64, Vec<MemoryEdge>> = HashMap::new();
    for _ in 0..n_edges {
        let from = read_u64(&mut f)?;
        let to = read_u64(&mut f)?;
        let mut weight = [0u8; 4];
        f.read_exact(&mut weight)?;
        edges.entry(from).or_default().push(MemoryEdge {
            to,
            weight: f32::from_le_bytes(weight),
        });
    }
    Ok((dim, bit_width, slot_to_id, records, edges))
}

fn write_u32(w: &mut impl Write, value: u32) -> io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_u64(w: &mut impl Write, value: u64) -> io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn write_i64(w: &mut impl Write, value: i64) -> io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_presence(r: &mut impl Read) -> io::Result<bool> {
    let mut value = [0u8; 1];
    r.read_exact(&mut value)?;
    match value[0] {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid optional-field marker {other}"),
        )),
    }
}

fn write_string(w: &mut impl Write, value: &str) -> io::Result<()> {
    let bytes = value.as_bytes();
    write_u32(w, bytes.len() as u32)?;
    w.write_all(bytes)
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u32(r)? as usize;
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}
