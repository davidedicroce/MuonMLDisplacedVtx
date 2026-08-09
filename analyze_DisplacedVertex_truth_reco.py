#!/usr/bin/env python3
"""
Truth-to-reconstructed-segment completeness study for displaced-muon ROOT files.

The ROOT files contain:
  * MuonVertexDump.truthMuonVertexMuonLinks:
      truth-muon indices attached to each truth vertex.
  * MuonBucketDump.segmentTruthPart:
      truth-particle index attached to each reconstructed segment.

For every displaced vertex with exactly two linked truth muons, this script
counts reconstructed segments belonging to each muon and labels the vertex
as 2/2, 1/2, or 0/2.  ROOT traversal is performed in compiled C++ so that
millions of MuonBucketDump entries are not iterated through Python.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


UNMATCHED_TRUTH_PART = 65535


def completeness_category(n_segments_muon0: int, n_segments_muon1: int) -> str:
    """Return the number of the two truth muons represented by >=1 segment."""
    represented = int(n_segments_muon0 > 0) + int(n_segments_muon1 > 0)
    return f"{represented}/2"


def _declare_root_helper():
    try:
        import ROOT
    except ImportError as exc:
        raise RuntimeError(
            "PyROOT is required. Activate the ROOT-enabled conda environment first."
        ) from exc

    if hasattr(ROOT, "DVTruthRecoStudy"):
        return ROOT

    # The production ROOT file stores the vertex-to-muon association as a
    # nested STL vector.  Some lightweight ROOT environments do not preload
    # its collection proxy, so request the dictionary before binding it.
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="dv_truth_reco_rootdict_") as dictionary_dir:
        try:
            os.chdir(dictionary_dir)
            ROOT.gInterpreter.GenerateDictionary(
                "vector<vector<unsigned short> >", "vector"
            )
        finally:
            os.chdir(original_directory)

    ok = ROOT.gInterpreter.Declare(
        r"""
#include <TBranch.h>
#include <TFile.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DVTruthRecoStudy {

struct EventKey {
  ULong64_t first{0};
  ULong64_t second{0};

  bool operator==(const EventKey& other) const {
    return first == other.first && second == other.second;
  }
};

struct EventKeyHash {
  std::size_t operator()(const EventKey& key) const {
    const std::size_t h1 = std::hash<ULong64_t>{}(key.first);
    const std::size_t h2 = std::hash<ULong64_t>{}(key.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
  }
};

struct BucketStats {
  ULong64_t nBuckets{0};
  ULong64_t nTruthBuckets{0};
  ULong64_t nSegments{0};
  ULong64_t nMatchedSegments{0};
  ULong64_t nUnmatchedSegments{0};
  ULong64_t nTruthVectorSizeMismatches{0};
  std::unordered_map<unsigned short, ULong64_t> segmentsPerTruthParticle{};
};

struct VertexRow {
  ULong64_t hash0{0};
  ULong64_t hash1{0};
  Long64_t eventEntry{-1};
  unsigned int vertexIndex{0};

  float xMm{0.F};
  float yMm{0.F};
  float zMm{0.F};
  float rMm{0.F};

  int muon0{-1};
  int muon1{-1};
  float muon0Pt{0.F};
  float muon0Eta{0.F};
  float muon0Phi{0.F};
  float muon1Pt{0.F};
  float muon1Eta{0.F};
  float muon1Phi{0.F};

  ULong64_t muon0Segments{0};
  ULong64_t muon1Segments{0};
  ULong64_t eventBuckets{0};
  ULong64_t eventTruthBuckets{0};
  ULong64_t eventSegments{0};
  ULong64_t eventMatchedSegments{0};
  ULong64_t eventUnmatchedSegments{0};
  ULong64_t eventTruthVectorSizeMismatches{0};
  bool linksValid{false};
};

struct FileResult {
  Long64_t bucketEntries{0};
  Long64_t vertexEntries{0};
  ULong64_t invalidTwoMuonLinks{0};
  std::vector<VertexRow> vertices{};
};

template <class T>
void requireBranch(TTree* tree, const char* name, T* address) {
  TBranch* branch = tree->GetBranch(name);
  if (!branch) {
    throw std::runtime_error(
        std::string{"Tree "} + tree->GetName() + " is missing branch " + name);
  }
  tree->SetBranchStatus(name, 1);
  if (branch->GetListOfBranches() && branch->GetListOfBranches()->GetEntries() > 0) {
    const std::string branchPattern = std::string{name} + "*";
    tree->SetBranchStatus(branchPattern.c_str(), 1);
  }
  if (tree->SetBranchAddress(name, address) < 0) {
    throw std::runtime_error(
        std::string{"Failed to bind branch "} + name + " in tree " + tree->GetName());
  }
}

template <class T>
void requireObjectBranch(TTree* tree, const char* name, T** address) {
  TBranch* branch = tree->GetBranch(name);
  if (!branch) {
    throw std::runtime_error(
        std::string{"Tree "} + tree->GetName() + " is missing branch " + name);
  }
  // These STL branches are split in the production TTrees.  Enabling only
  // the parent name leaves its data sub-branches disabled and produces null
  // collection pointers, so enable the complete branch family before binding.
  tree->SetBranchStatus(name, 1);
  if (branch->GetListOfBranches() && branch->GetListOfBranches()->GetEntries() > 0) {
    const std::string branchPattern = std::string{name} + "*";
    tree->SetBranchStatus(branchPattern.c_str(), 1);
  }
  if (tree->SetBranchAddress(name, address) < 0) {
    throw std::runtime_error(
        std::string{"Failed to bind branch "} + name + " in tree " + tree->GetName());
  }
}

FileResult analyzeFile(const std::string& fileName, const double minDisplacedRadiusMm) {
  std::unique_ptr<TFile> input{TFile::Open(fileName.c_str(), "READ")};
  if (!input || input->IsZombie()) {
    throw std::runtime_error("Could not open ROOT file: " + fileName);
  }

  auto* bucketTree = dynamic_cast<TTree*>(input->Get("MuonBucketDump"));
  auto* vertexTree = dynamic_cast<TTree*>(input->Get("MuonVertexDump"));
  if (!bucketTree || !vertexTree) {
    throw std::runtime_error(
        "Expected MuonBucketDump and MuonVertexDump trees in " + fileName);
  }

  FileResult result{};
  result.bucketEntries = bucketTree->GetEntries();
  result.vertexEntries = vertexTree->GetEntries();

  std::unordered_map<EventKey, BucketStats, EventKeyHash> eventStats{};
  eventStats.reserve(static_cast<std::size_t>(
      std::max<Long64_t>(result.vertexEntries * 2, 1024)));

  bucketTree->SetBranchStatus("*", 0);
  ULong64_t bucketHash[2]{0, 0};
  unsigned char bucketHasTruth{0};
  std::vector<unsigned short>* segmentTruthPart{nullptr};
  std::vector<float>* segmentPositionX{nullptr};

  requireBranch(bucketTree, "CommonEventHash[2]/l", bucketHash);
  requireBranch(bucketTree, "bucket_hasTruth", &bucketHasTruth);
  requireObjectBranch(bucketTree, "segmentTruthPart", &segmentTruthPart);
  requireObjectBranch(bucketTree, "segmentPositionX", &segmentPositionX);

  for (Long64_t entry = 0; entry < result.bucketEntries; ++entry) {
    if (bucketTree->GetEntry(entry) <= 0) continue;

    BucketStats& stats = eventStats[EventKey{bucketHash[0], bucketHash[1]}];
    ++stats.nBuckets;
    stats.nTruthBuckets += static_cast<ULong64_t>(bucketHasTruth != 0);

    const std::size_t nSegments =
        segmentPositionX ? segmentPositionX->size() : std::size_t{0};
    const std::size_t nTruthIds =
        segmentTruthPart ? segmentTruthPart->size() : std::size_t{0};
    stats.nSegments += static_cast<ULong64_t>(nSegments);
    stats.nTruthVectorSizeMismatches +=
        static_cast<ULong64_t>(nSegments != nTruthIds);

    for (std::size_t idx = 0; idx < nSegments; ++idx) {
      const unsigned short truthId =
          idx < nTruthIds ? (*segmentTruthPart)[idx]
                          : static_cast<unsigned short>(65535);
      if (truthId == static_cast<unsigned short>(65535)) {
        ++stats.nUnmatchedSegments;
      } else {
        ++stats.nMatchedSegments;
        ++stats.segmentsPerTruthParticle[truthId];
      }
    }
  }

  vertexTree->SetBranchStatus("*", 0);
  ULong64_t vertexHash[2]{0, 0};
  std::vector<float>* vertexX{nullptr};
  std::vector<float>* vertexY{nullptr};
  std::vector<float>* vertexZ{nullptr};
  std::vector<std::vector<unsigned short>>* vertexMuonLinks{nullptr};
  std::vector<float>* truthMuonPt{nullptr};
  std::vector<float>* truthMuonEta{nullptr};
  std::vector<float>* truthMuonPhi{nullptr};

  requireBranch(vertexTree, "CommonEventHash[2]/l", vertexHash);
  requireObjectBranch(vertexTree, "truthMuonVertexPositionX", &vertexX);
  requireObjectBranch(vertexTree, "truthMuonVertexPositionY", &vertexY);
  requireObjectBranch(vertexTree, "truthMuonVertexPositionZ", &vertexZ);
  requireObjectBranch(vertexTree, "truthMuonVertexMuonLinks", &vertexMuonLinks);
  requireObjectBranch(vertexTree, "truthMuon_pt", &truthMuonPt);
  requireObjectBranch(vertexTree, "truthMuon_eta", &truthMuonEta);
  requireObjectBranch(vertexTree, "truthMuon_phi", &truthMuonPhi);

  for (Long64_t entry = 0; entry < result.vertexEntries; ++entry) {
    if (vertexTree->GetEntry(entry) <= 0) continue;
    if (!vertexX || !vertexY || !vertexZ || !vertexMuonLinks ||
        !truthMuonPt || !truthMuonEta || !truthMuonPhi) {
      continue;
    }

    const std::size_t nVertices = std::min(
        {vertexX->size(), vertexY->size(), vertexZ->size(), vertexMuonLinks->size()});
    const EventKey eventKey{vertexHash[0], vertexHash[1]};
    const auto statsItr = eventStats.find(eventKey);
    const BucketStats emptyStats{};
    const BucketStats& stats =
        statsItr == eventStats.end() ? emptyStats : statsItr->second;

    for (std::size_t iv = 0; iv < nVertices; ++iv) {
      const float x = (*vertexX)[iv];
      const float y = (*vertexY)[iv];
      const float z = (*vertexZ)[iv];
      const float radius = std::hypot(x, y);
      const auto& links = (*vertexMuonLinks)[iv];

      // The primary interaction vertex is near the beam line.  This study
      // only considers displaced vertices that decay to exactly two muons.
      if (radius <= minDisplacedRadiusMm || links.size() != 2U) continue;

      VertexRow row{};
      row.hash0 = vertexHash[0];
      row.hash1 = vertexHash[1];
      row.eventEntry = entry;
      row.vertexIndex = static_cast<unsigned int>(iv);
      row.xMm = x;
      row.yMm = y;
      row.zMm = z;
      row.rMm = radius;
      row.muon0 = static_cast<int>(links[0]);
      row.muon1 = static_cast<int>(links[1]);

      const auto indexIsValid = [&](const int index) {
        return index >= 0 &&
               static_cast<std::size_t>(index) < truthMuonPt->size() &&
               static_cast<std::size_t>(index) < truthMuonEta->size() &&
               static_cast<std::size_t>(index) < truthMuonPhi->size();
      };
      row.linksValid = indexIsValid(row.muon0) && indexIsValid(row.muon1);
      if (!row.linksValid) {
        ++result.invalidTwoMuonLinks;
      } else {
        row.muon0Pt = (*truthMuonPt)[row.muon0];
        row.muon0Eta = (*truthMuonEta)[row.muon0];
        row.muon0Phi = (*truthMuonPhi)[row.muon0];
        row.muon1Pt = (*truthMuonPt)[row.muon1];
        row.muon1Eta = (*truthMuonEta)[row.muon1];
        row.muon1Phi = (*truthMuonPhi)[row.muon1];
      }

      const auto segmentCount = [&](const int truthId) -> ULong64_t {
        if (truthId < 0) return 0;
        const auto itr = stats.segmentsPerTruthParticle.find(
            static_cast<unsigned short>(truthId));
        return itr == stats.segmentsPerTruthParticle.end() ? 0 : itr->second;
      };
      row.muon0Segments = segmentCount(row.muon0);
      row.muon1Segments = segmentCount(row.muon1);
      row.eventBuckets = stats.nBuckets;
      row.eventTruthBuckets = stats.nTruthBuckets;
      row.eventSegments = stats.nSegments;
      row.eventMatchedSegments = stats.nMatchedSegments;
      row.eventUnmatchedSegments = stats.nUnmatchedSegments;
      row.eventTruthVectorSizeMismatches = stats.nTruthVectorSizeMismatches;
      result.vertices.push_back(row);
    }
  }

  return result;
}

}  // namespace DVTruthRecoStudy
"""
    )
    if not ok:
        raise RuntimeError(
            "ROOT could not compile the truth/reconstruction extraction helper. "
            "Activate the ROOT-enabled conda environment."
        )
    return ROOT


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def _atomic_json(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)


def _vertex_records(
    source_file: Path,
    result,
    envelope_r_max_mm: float,
    envelope_z_max_mm: float,
):
    records = []
    for row in result.vertices:
        muon0_segments = int(row.muon0Segments)
        muon1_segments = int(row.muon1Segments)
        records.append(
            {
                "source_file": source_file.name,
                "event_hash0": int(row.hash0),
                "event_hash1": int(row.hash1),
                "event_entry": int(row.eventEntry),
                "vertex_index": int(row.vertexIndex),
                "vertex_x_mm": float(row.xMm),
                "vertex_y_mm": float(row.yMm),
                "vertex_z_mm": float(row.zMm),
                "vertex_r_mm": float(row.rMm),
                "inside_calo_envelope": bool(
                    float(row.rMm) <= envelope_r_max_mm
                    and abs(float(row.zMm)) <= envelope_z_max_mm
                ),
                "links_valid": bool(row.linksValid),
                "muon0_index": int(row.muon0),
                "muon1_index": int(row.muon1),
                "muon0_pt": float(row.muon0Pt),
                "muon0_eta": float(row.muon0Eta),
                "muon0_phi": float(row.muon0Phi),
                "muon1_pt": float(row.muon1Pt),
                "muon1_eta": float(row.muon1Eta),
                "muon1_phi": float(row.muon1Phi),
                "muon0_segments": muon0_segments,
                "muon1_segments": muon1_segments,
                "completeness_category": completeness_category(
                    muon0_segments, muon1_segments
                ),
                "event_buckets": int(row.eventBuckets),
                "event_truth_buckets": int(row.eventTruthBuckets),
                "event_segments": int(row.eventSegments),
                "event_matched_segments": int(row.eventMatchedSegments),
                "event_unmatched_segments": int(row.eventUnmatchedSegments),
                "truth_vector_size_mismatches": int(
                    row.eventTruthVectorSizeMismatches
                ),
            }
        )
    return records


def _build_event_table(vertices: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_file",
        "event_hash0",
        "event_hash1",
        "event_entry",
        "n_displaced_vertices",
        "n_calo_envelope_vertices",
        "n_linked_truth_muons",
        "n_represented_truth_muons",
        "n_complete_vertices",
        "event_completeness_category",
        "event_buckets",
        "event_truth_buckets",
        "event_segments",
        "event_matched_segments",
        "event_unmatched_segments",
        "truth_vector_size_mismatches",
    ]
    if vertices.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    keys = ["source_file", "event_hash0", "event_hash1", "event_entry"]
    for key, group in vertices.groupby(keys, sort=False, dropna=False):
        linked_muons: dict[int, int] = {}
        for item in group.itertuples(index=False):
            linked_muons[item.muon0_index] = max(
                linked_muons.get(item.muon0_index, 0), item.muon0_segments
            )
            linked_muons[item.muon1_index] = max(
                linked_muons.get(item.muon1_index, 0), item.muon1_segments
            )
        n_linked = len(linked_muons)
        n_represented = sum(count > 0 for count in linked_muons.values())
        first = group.iloc[0]
        rows.append(
            {
                "source_file": key[0],
                "event_hash0": int(key[1]),
                "event_hash1": int(key[2]),
                "event_entry": int(key[3]),
                "n_displaced_vertices": int(len(group)),
                "n_calo_envelope_vertices": int(
                    group["inside_calo_envelope"].sum()
                ),
                "n_linked_truth_muons": int(n_linked),
                "n_represented_truth_muons": int(n_represented),
                "n_complete_vertices": int(
                    (group["completeness_category"] == "2/2").sum()
                ),
                "event_completeness_category": f"{n_represented}/{n_linked}",
                "event_buckets": int(first["event_buckets"]),
                "event_truth_buckets": int(first["event_truth_buckets"]),
                "event_segments": int(first["event_segments"]),
                "event_matched_segments": int(first["event_matched_segments"]),
                "event_unmatched_segments": int(first["event_unmatched_segments"]),
                "truth_vector_size_mismatches": int(
                    first["truth_vector_size_mismatches"]
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


def _summary_payload(
    source_file: Path,
    result,
    vertices: pd.DataFrame,
    events: pd.DataFrame,
) -> dict:
    category_counts = {
        category: int((vertices["completeness_category"] == category).sum())
        if not vertices.empty
        else 0
        for category in ("2/2", "1/2", "0/2")
    }
    inside_envelope = (
        vertices[vertices["inside_calo_envelope"]]
        if not vertices.empty
        else vertices
    )
    envelope_counts = {
        category: int(
            (inside_envelope["completeness_category"] == category).sum()
        )
        if not inside_envelope.empty
        else 0
        for category in ("2/2", "1/2", "0/2")
    }
    denominator = len(vertices)
    envelope_denominator = len(inside_envelope)
    return {
        "source_file": str(source_file),
        "bucket_entries": int(result.bucketEntries),
        "vertex_entries": int(result.vertexEntries),
        "displaced_vertices": denominator,
        "events_with_displaced_vertices": int(len(events)),
        "invalid_two_muon_links": int(result.invalidTwoMuonLinks),
        "truth_vector_size_mismatches": int(
            vertices["truth_vector_size_mismatches"].sum()
            if not vertices.empty
            else 0
        ),
        "category_counts": category_counts,
        "category_fractions": {
            key: (value / denominator if denominator else 0.0)
            for key, value in category_counts.items()
        },
        "calo_envelope_displaced_vertices": envelope_denominator,
        "calo_envelope_category_counts": envelope_counts,
        "calo_envelope_category_fractions": {
            key: (value / envelope_denominator if envelope_denominator else 0.0)
            for key, value in envelope_counts.items()
        },
    }


def analyze_one_file(
    source_file: Path,
    output_dir: Path,
    *,
    r_min_mm: float,
    envelope_r_max_mm: float,
    envelope_z_max_mm: float,
    overwrite: bool,
) -> dict:
    stem = source_file.stem
    vertex_output = output_dir / f"{stem}.vertices.parquet"
    event_output = output_dir / f"{stem}.events.parquet"
    summary_output = output_dir / f"{stem}.summary.json"

    if (
        not overwrite
        and vertex_output.exists()
        and event_output.exists()
        and summary_output.exists()
    ):
        with summary_output.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        print(f"[skip] completed outputs already exist for {source_file.name}")
        return summary

    ROOT = _declare_root_helper()
    print(f"[read] {source_file}")
    result = ROOT.DVTruthRecoStudy.analyzeFile(str(source_file), float(r_min_mm))
    vertices = pd.DataFrame.from_records(
        _vertex_records(
            source_file,
            result,
            envelope_r_max_mm,
            envelope_z_max_mm,
        )
    )
    events = _build_event_table(vertices)
    summary = _summary_payload(source_file, result, vertices, events)

    _atomic_parquet(vertices, vertex_output)
    _atomic_parquet(events, event_output)
    _atomic_json(summary, summary_output)

    counts = summary["category_counts"]
    print(
        f"[done] events={len(events)} vertices={len(vertices)} "
        f"2/2={counts['2/2']} 1/2={counts['1/2']} 0/2={counts['0/2']}"
    )
    return summary


def _resolve_inputs(args) -> list[Path]:
    paths: list[Path] = []
    for value in args.input_file:
        paths.append(Path(value).expanduser())
    if args.input_dir:
        paths.extend(sorted(Path(args.input_dir).expanduser().glob(args.glob)))

    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if args.max_files is not None:
        unique = unique[: args.max_files]
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input ROOT files do not exist: {missing}")
    if not unique:
        raise ValueError("No ROOT input files were selected.")
    return unique


def _self_test() -> None:
    assert completeness_category(3, 2) == "2/2"
    assert completeness_category(3, 0) == "1/2"
    assert completeness_category(0, 4) == "1/2"
    assert completeness_category(0, 0) == "0/2"

    synthetic = pd.DataFrame.from_records(
        [
            {
                "source_file": "sample.root",
                "event_hash0": 1,
                "event_hash1": 2,
                "event_entry": 0,
                "inside_calo_envelope": True,
                "muon0_index": 0,
                "muon1_index": 1,
                "muon0_segments": 3,
                "muon1_segments": 0,
                "completeness_category": "1/2",
                "event_buckets": 10,
                "event_truth_buckets": 5,
                "event_segments": 3,
                "event_matched_segments": 3,
                "event_unmatched_segments": 0,
                "truth_vector_size_mismatches": 0,
            }
        ]
    )
    events = _build_event_table(synthetic)
    assert len(events) == 1
    assert events.iloc[0]["event_completeness_category"] == "1/2"
    assert events.iloc[0]["n_complete_vertices"] == 0
    print("[self-test] passed")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file",
        action="append",
        default=[],
        help="ROOT file to analyze; may be supplied multiple times.",
    )
    parser.add_argument("--input-dir", help="Directory containing ROOT files.")
    parser.add_argument("--glob", default="*.root", help="Glob used with --input-dir.")
    parser.add_argument("--max-files", type=int, help="Analyze at most this many files.")
    parser.add_argument("--output-dir", help="Directory for Parquet and JSON results.")
    parser.add_argument("--r-min-mm", type=float, default=30.0)
    parser.add_argument(
        "--envelope-r-max-mm",
        "--r-max-mm",
        dest="envelope_r_max_mm",
        type=float,
        default=4250.0,
        help=(
            "Radius of the simplified converter calorimeter envelope; "
            "--r-max-mm is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--envelope-z-max-mm",
        "--z-max-mm",
        dest="envelope_z_max_mm",
        type=float,
        default=6500.0,
        help=(
            "Absolute-z extent of the simplified converter calorimeter envelope; "
            "--z-max-mm is retained as a compatibility alias."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.output_dir:
        raise ValueError("--output-dir is required unless --self-test is used.")

    inputs = _resolve_inputs(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for source_file in inputs:
        summaries.append(
            analyze_one_file(
                source_file,
                output_dir,
                r_min_mm=args.r_min_mm,
                envelope_r_max_mm=args.envelope_r_max_mm,
                envelope_z_max_mm=args.envelope_z_max_mm,
                overwrite=args.overwrite,
            )
        )

    combined = {
        "files": len(summaries),
        "displaced_vertices": sum(item["displaced_vertices"] for item in summaries),
        "events_with_displaced_vertices": sum(
            item["events_with_displaced_vertices"] for item in summaries
        ),
        "category_counts": {
            category: sum(
                item["category_counts"][category] for item in summaries
            )
            for category in ("2/2", "1/2", "0/2")
        },
    }
    total = combined["displaced_vertices"]
    combined["category_fractions"] = {
        key: (value / total if total else 0.0)
        for key, value in combined["category_counts"].items()
    }
    _atomic_json(combined, output_dir / "combined_summary.json")
    print(json.dumps(combined, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
