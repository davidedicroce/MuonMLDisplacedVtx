#!/usr/bin/env python3
"""Evaluate a supplied DisplacedVertex ONNX graph classifier for the existing plotting notebook.

This script DOES NOT need a PyTorch checkpoint and DOES NOT re-export the model.
It reuses the H5 reader, split logic, histogram metrics and plotting-payload
builders in DisplacedVertex_results_tune.py to ensure metric compatibility.

Example (append a collaborator's model to your existing comparison results):
    python DisplacedVertex_results_single.py \
        --onnx edge_gatv2.onnx \
        --results-script DisplacedVertex_results_tune.py \
        --data-glob '/shared/wp2p5/data/data_displacedVtx_mu200_graphs/*.h5' \
        --split-file /shared/wp2p5/data/data_displacedVtx_mu200_graphs/split_displaced_vertex_seed12345.npz \
        --output-dir onnx_eval_compare_displaced_vertex

The notebook should then use OUTDIR = Path('onnx_eval_compare_displaced_vertex').
If this directory already contains old-model results, its tables and payload
are UPDATED (upserted), preserving the existing comparison models.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

DEFAULT_RESULTS_SCRIPT_NAMES = (
    'DisplacedVertex_results_tune.py',
    'DisplacedVertex_results_tune(20261019-102842).py',
)


def load_original_evaluator(path: Path):
    """Import the provided evaluator, without executing its main()."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Original evaluation script not found: {path}')
    # Make dv_training_utils.py importable when it lives alongside the evaluator.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('dv_original_evaluator_for_external_onnx', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Unable to import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f'Could not import original evaluation utilities: {exc}. '
            'Place dv_training_utils.py (the module used by your original script) '
            'next to the original evaluation script, and install its dependencies.'
        ) from exc
    return mod


def default_results_script() -> Path:
    here = Path(__file__).resolve().parent
    for name in DEFAULT_RESULTS_SCRIPT_NAMES:
        p = here / name
        if p.is_file():
            return p
    return here / DEFAULT_RESULTS_SCRIPT_NAMES[0]


def parse_args():
    ap = argparse.ArgumentParser(
        description='Evaluate an existing DisplacedVertex ONNX model and update the original notebook inputs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--onnx', type=Path, required=True, help='Already-exported ONNX model, e.g. edge_gatv2.onnx')
    ap.add_argument('--results-script', type=Path, default=default_results_script(),
                    help='Original DisplacedVertex_results_tune.py (reuses identical evaluation/metrics helpers).')
    ap.add_argument('--data-glob', default=None, help='Full H5 set, same glob/order used to construct the split file.')
    ap.add_argument('--split-file', type=Path, default=None, help='Original train/validation split .npz.')
    ap.add_argument('--split-key', default='val_idx', help='Held-out split key; use all only for explicitly intended full-dataset evaluation.')
    ap.add_argument('--strict-split-check', action='store_true', help='Require strict file-path checks against the saved split.')
    ap.add_argument('--output-dir', type=Path, default=Path('onnx_eval_compare_displaced_vertex'),
                    help='Existing results directory to append to, or a new directory for just this model.')
    ap.add_argument('--display-name', default=None, help='Label displayed in plots (default: ONNX filename without extension).')
    ap.add_argument('--run-group', default='external_onnx', help='Grouping label in the payload/notebook.')
    ap.add_argument('--model-id', default=None, help='Optional stable identifier; default uses the ONNX SHA-256 hash.')
    ap.add_argument('--single-output-mode', choices=['logit', 'prob'], default='logit',
                    help='logit -> sigmoid(output); edge_gatv2.onnx is a logit model.')
    ap.add_argument('--no-cuda', action='store_true', help='Use CPUExecutionProvider even if CUDA is available.')
    ap.add_argument('--ort-intra-threads', type=int, default=1)
    ap.add_argument('--ort-inter-threads', type=int, default=1)
    ap.add_argument('--metric-bins', type=int, default=None, help='Histogram bins. Inherited from old payload when present.')
    ap.add_argument('--target-fprs', nargs='+', type=float, default=None, help='Inherit from old payload or use 0.001 0.005 0.01 0.02.')
    ap.add_argument('--target-tprs', nargs='+', type=float, default=None, help='Inherit from old payload or use 0.90 0.95 0.99.')
    ap.add_argument('--partial-auc-fpr-max', nargs='+', type=float, default=None, help='Inherit from old payload or use 0.001 0.005 0.01 0.02.')
    ap.add_argument('--plot-threshold', type=float, default=None, help='Default confusion-matrix cut; inherited if possible (else 0.5).')
    ap.add_argument('--max-val-events', type=int, default=None, help='For quick tests only; omit for real comparison.')
    ap.add_argument('--print-every', type=int, default=10000)
    return ap.parse_args()


def existing_payload_and_options(mod, args, output_dir: Path):
    path = output_dir / 'plot_payload.pkl'
    payload = pd.read_pickle(path) if path.exists() else {}
    if payload and not isinstance(payload, dict):
        raise TypeError(f'Expected a dict in {path}')

    options = {
        'metric_bins': (args.metric_bins, payload.get('metric_bins'), 200000),
        'target_fprs': (args.target_fprs, payload.get('target_fprs'), [0.001, 0.005, 0.01, 0.02]),
        'target_tprs': (args.target_tprs, payload.get('target_tprs'), [0.90, 0.95, 0.99]),
        'partial_auc_fpr_max': (args.partial_auc_fpr_max, payload.get('partial_auc_fpr_max'), [0.001, 0.005, 0.01, 0.02]),
        'plot_threshold': (args.plot_threshold, payload.get('plot_threshold'), 0.5),
    }
    config = {k: next((v for v in triplet if v is not None), None) for k, triplet in options.items()}
    config['metric_bins'] = int(config['metric_bins'])
    config['plot_threshold'] = float(config['plot_threshold'])
    for key in ('target_fprs', 'target_tprs', 'partial_auc_fpr_max'):
        config[key] = [float(v) for v in config[key]]
    if config['metric_bins'] < 2:
        raise ValueError('metric-bins must be >= 2')
    if payload:
        for key, (_, stored, _) in options.items():
            if stored is None:
                continue
            if isinstance(stored, (list, tuple, np.ndarray)):
                same = (len(config[key]) == len(stored)
                        and np.allclose(config[key], stored, rtol=0, atol=1e-12))
            else:
                same = bool(np.isclose(config[key], stored, rtol=0, atol=1e-12))
            if not same:
                raise ValueError(
                    f'{key} differs from existing plot_payload.pkl: {config[key]} versus {stored}. '
                    'Use the same settings as the original models or a NEW --output-dir.'
                )
    return payload, config


def model_metadata_from_session(model_path: Path, mod, no_cuda: bool, intra_threads: int, inter_threads: int):
    sess = mod.make_ort_session(model_path, no_cuda, intra_threads, inter_threads)
    inputs = {i.name: i for i in sess.get_inputs()}
    expected = {
        'x': 'tensor(float)',
        'edge_index': 'tensor(int64)',
        'edge_attr': 'tensor(float)',
        'n_muon_nodes': 'tensor(int64)',
    }
    if set(inputs) != set(expected):
        raise ValueError(f'ONNX inputs {list(inputs)} differ from the expected {list(expected)}. '
                         'This script targets the DisplacedVertex graph-input ONNX contract.')
    for name, dtype in expected.items():
        if inputs[name].type != dtype:
            raise ValueError(f'ONNX input {name} has dtype {inputs[name].type}; expected {dtype}')
    x_shape, e_shape, ei_shape = inputs['x'].shape, inputs['edge_attr'].shape, inputs['edge_index'].shape
    if len(x_shape) != 2 or x_shape[1] != 7 or len(e_shape) != 2 or e_shape[1] != 5 or ei_shape[0] != 2:
        raise ValueError(f'Unexpected ONNX feature dimensions: x={x_shape}, edge_attr={e_shape}, edge_index={ei_shape}')
    outs = sess.get_outputs()
    if len(outs) != 1:
        raise ValueError(f'Expected one logit output, found {[o.name for o in outs]}')
    metadata = dict(sess.get_modelmeta().custom_metadata_map or {})
    print('[onnx] Inputs:', [(i.name, i.shape, i.type) for i in inputs.values()], flush=True)
    print('[onnx] Output:', [(o.name, o.shape, o.type) for o in outs], flush=True)
    print('[onnx] Embedded normalization: node=', metadata.get('normalize_node_features'),
          ', edge=', metadata.get('normalize_edge_features'), flush=True)
    print('[onnx] Inference provider(s):', sess.get_providers(), flush=True)
    del sess
    return metadata


def check_old_run_compatible(old_run: dict, input_paths: list[Path], split_file: Path | None, split_key: str, max_events):
    """Prevent comparing models evaluated on different events under one ranking."""
    if not old_run:
        return
    if old_run.get('split_key') is not None and old_run['split_key'] != split_key:
        raise ValueError(f'Old results use split_key={old_run["split_key"]!r} but requested {split_key!r}. '
                         'Use the same split or a new --output-dir.')
    if old_run.get('split_file') and split_file and Path(old_run['split_file']).resolve() != split_file.resolve():
        raise ValueError('Old results used a different --split-file. Use the same split or a new --output-dir.')
    if old_run.get('input_files'):
        old_paths = [str(Path(p).resolve()) for p in old_run['input_files']]
        new_paths = [str(p.resolve()) for p in input_paths]
        if old_paths != new_paths:
            raise ValueError('Old results used a different ordered H5 file list. '
                             'Use the same --data-glob or a new --output-dir.')
    if max_events is not None and old_run.get('num_validation_events') is not None:
        # Can't compare restricted new evaluation with larger existing evaluation.
        if int(max_events) < int(old_run['num_validation_events']):
            raise ValueError('A --max-val-events subset is not comparable to the existing full validation result. '
                             'Use a new --output-dir for a smoke test.')


def main():
    args = parse_args()
    onnx_path = args.onnx.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f'ONNX file not found: {onnx_path}')
    mod = load_original_evaluator(args.results_script)
    if mod.ort is None:
        raise RuntimeError('onnxruntime is not installed: pip install onnxruntime (or onnxruntime-gpu).')

    data_glob = args.data_glob or mod.DEFAULT_DATA_GLOB
    split_file = (args.split_file if args.split_file is not None else mod.DEFAULT_SPLIT_FILE)
    split_file = split_file.expanduser().resolve() if split_file is not None else None
    if args.split_key != 'all' and (split_file is None or not split_file.is_file()):
        raise FileNotFoundError(f'Split file is required for {args.split_key}: {split_file}')
    input_paths = mod.resolve_input_paths(data_glob)  # CRITICAL: entire split-compatible ordered H5 list

    output_dir = args.output_dir.expanduser().resolve()
    # No writing occurs before the comparisons and dependency checks complete.
    old_run_path = output_dir / 'run_summary.json'
    old_run = json.loads(old_run_path.read_text(encoding='utf-8')) if old_run_path.exists() else {}
    check_old_run_compatible(old_run, input_paths, split_file, args.split_key, args.max_val_events)
    payload, cfg = existing_payload_and_options(mod, args, output_dir)
    metadata = model_metadata_from_session(onnx_path, mod, args.no_cuda,
                                            args.ort_intra_threads, args.ort_inter_threads)
    declared_mode = metadata.get('single_output_mode')
    if declared_mode in ('logit', 'prob') and declared_mode != args.single_output_mode:
        raise ValueError(f'ONNX metadata declares single_output_mode={declared_mode!r}; '
                         f'you requested {args.single_output_mode!r}.')

    # A metadata shim is used ONLY for logging; all input normalization is already in ONNX.
    # The imported evaluate_onnx_model_on_files() sends raw H5 x/edge_attr to ONNX.
    ckpt_metadata = {}
    for name in ('normalize_node_features', 'normalize_edge_features'):
        if name in metadata:
            ckpt_metadata[name] = str(metadata[name]).strip().lower() == 'true'
    for name in ('feature_norm_kind', 'feature_stats_json'):
        if name in metadata:
            ckpt_metadata[name] = metadata[name]
    if 'feature_norm_clip' in metadata:
        ckpt_metadata['feature_norm_clip'] = float(metadata['feature_norm_clip'])

    eval_args = SimpleNamespace(
        split_file=split_file,
        split_key=args.split_key,
        strict_split_check=args.strict_split_check,
        max_val_events=args.max_val_events,
        feature_stats_json=None,
        feature_norm_kind=None,
        normalize_node_features=None,
        normalize_edge_features=None,
        feature_norm_clip=None,
        print_every=args.print_every,
    )
    print(f'[data] {len(input_paths)} H5 files; split={args.split_key!r}; full dataset glob={data_glob}', flush=True)
    hist_state, latencies, providers, eval_meta = mod.evaluate_onnx_model_on_files(
        onnx_path, input_paths,
        ckpt=ckpt_metadata,
        args=eval_args,
        no_cuda=args.no_cuda,
        intra_threads=args.ort_intra_threads,
        inter_threads=args.ort_inter_threads,
        single_output_mode=args.single_output_mode,
        metric_bins=cfg['metric_bins'],
    )
    if old_run.get('num_validation_events') and int(old_run['num_validation_events']) != int(len(latencies)):
        raise ValueError('New evaluation has a different number of validation events from the existing results. '
                         'Use a new --output-dir to avoid an invalid comparison.')

    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    model_id = args.model_id or f'external_onnx_sha256:{digest}'
    display_name = args.display_name or onnx_path.stem
    layer_type = metadata.get('layer_type', 'external_onnx')
    sample_names = sorted(hist_state['sample_hists'])

    inspect = mod.inspect_onnx_artifact(onnx_path)
    export_info = {
        'model_id': model_id,
        'display_name': display_name,
        'run_group': args.run_group,
        'onnx_path': str(onnx_path),
        'exported': True,
        'reason': 'External ONNX supplied; no export performed',
        'onnx_model_size_mb': inspect['onnx_model_size_mb'],
        'onnx_initializer_elements': inspect['onnx_initializer_elements'],
    }
    summary_rows, tpr_rows, fpr_rows, pauc_rows, eff_rows = [], [], [], [], []
    mod.append_metric_rows_from_hist(
        summary_rows=summary_rows,
        op_tpr_rows=tpr_rows,
        op_fpr_rows=fpr_rows,
        pauc_fpr_rows=pauc_rows,
        target_tprs=cfg['target_tprs'],
        target_fprs=cfg['target_fprs'],
        partial_auc_fpr_max=cfg['partial_auc_fpr_max'],
        model_id=model_id,
        display_name=display_name,
        run_group=args.run_group,
        layer_type=layer_type,
        ckpt_path='',
        threshold_ckpt=cfg['plot_threshold'],
        target_fpr_ckpt=np.nan,
        best_monitor=None,
        hist_state=hist_state,
        lat_ort=latencies,
        providers=providers,
        export_info=export_info,
        eval_meta=eval_meta,
        metrics_source='onnxruntime_hist',
    )
    mod.append_signal_efficiency_rows_from_hist(
        signal_efficiency_rows=eff_rows,
        target_fprs=cfg['target_fprs'],
        model_id=model_id,
        display_name=display_name,
        run_group=args.run_group,
        layer_type=layer_type,
        hist_state=hist_state,
        metrics_source='onnxruntime_hist',
    )

    model_entry = mod.plot_model_payload_from_hist(
        model_id=model_id,
        display_name=display_name,
        run_group=args.run_group,
        layer_type=layer_type,
        ckpt_path='',
        onnx_path=str(onnx_path),
        hist_state=hist_state,
        lat_ort=latencies,
        providers=providers,
        target_tprs=cfg['target_tprs'],
        target_fprs=cfg['target_fprs'],
        default_threshold=cfg['plot_threshold'],
        include_threshold_scan=True,
    )

    # Same filenames, column schemas, and keys as the original plotting notebook.
    output_dir.mkdir(parents=True, exist_ok=True)
    table_updates = (
        ('model_summary.csv', summary_rows, ['model_id']),
        ('accuracy_at_target_tpr.csv', tpr_rows, ['model_id', 'target_tpr']),
        ('tpr_at_target_fpr.csv', fpr_rows, ['model_id', 'target_fpr']),
        ('partial_auc_low_fpr.csv', pauc_rows, ['model_id', 'fpr_max']),
        ('signal_efficiency_by_sample_at_target_fpr.csv', eff_rows, ['model_id', 'sample', 'target_fpr']),
        ('onnx_export_status.csv', [export_info], ['model_id']),
    )
    for filename, rows, keys in table_updates:
        csv_path = output_dir / filename
        existing = mod.load_existing_table(csv_path)
        combined = mod.upsert_rows(existing, rows, keys)
        if not combined.empty:
            combined.to_csv(csv_path, index=False)
            print(f'[saved] {csv_path} ({len(combined)} rows)', flush=True)

    if not payload:
        signal_names = mod.sample_names_from_patterns(mod.DEFAULT_SIGNAL_DATA_GLOBS)
        background_names = mod.sample_names_from_patterns(mod.DEFAULT_BACKGROUND_DATA_GLOBS)
        payload = {
            'format': 'displaced_vertex_results_tune_plot_payload_v1',
            'description': 'ONNX Runtime histogram-derived DisplacedVertex plotting payload.',
            'target_tprs': cfg['target_tprs'],
            'target_fprs': cfg['target_fprs'],
            'partial_auc_fpr_max': cfg['partial_auc_fpr_max'],
            'signal_samples': signal_names,
            'background_samples': background_names,
            'best_model_detail_samples': list(dict.fromkeys(['__all__', *signal_names])),
            'evaluation_input_files_are_from_full_data_glob': True,
            'explicit_sample_input_files': [],
            'significance_definition': 'S/sqrt(S+B), cumulative validation counts above score threshold',
            'signal_efficiency_definition': 'Per-signal-sample validation efficiencies at globally chosen FPR thresholds',
            'metric_bins': cfg['metric_bins'],
            'plot_threshold': cfg['plot_threshold'],
            'contains_raw_scores': False,
            'contains_raw_labels': False,
        }
    entries = {str(e['model_id']): e for e in payload.get('models', [])}
    entries[model_id] = model_entry
    payload['models'] = sorted(entries.values(), key=lambda e: (
        str(e.get('run_group', '')), str(e.get('display_name', ''))))
    pd.to_pickle(payload, output_dir / 'plot_payload.pkl')
    print(f'[saved] {output_dir / "plot_payload.pkl"} ({len(entries)} total models)', flush=True)

    run = dict(old_run)
    run.setdefault('data_glob', data_glob)
    run.setdefault('input_files', [str(p) for p in input_paths])
    run.setdefault('split_file', str(split_file) if split_file is not None else None)
    run.setdefault('split_key', args.split_key)
    run.setdefault('target_fprs', cfg['target_fprs'])
    run.setdefault('target_tprs', cfg['target_tprs'])
    run.setdefault('partial_auc_fpr_max', cfg['partial_auc_fpr_max'])
    run.setdefault('metric_bins', cfg['metric_bins'])
    run.setdefault('plot_threshold', cfg['plot_threshold'])
    run.setdefault('num_validation_events', len(latencies))
    run['output_dir'] = str(output_dir)
    run['plot_payload_path'] = str(output_dir / 'plot_payload.pkl')
    run['num_plot_payload_models'] = len(entries)
    external_evaluations = {e['model_id']: e for e in run.get('external_onnx_evaluations', [])}
    external_evaluations[model_id] = {
        'model_id': model_id,
        'display_name': display_name,
        'onnx_path': str(onnx_path),
        'onnx_sha256': digest,
        'onnx_metadata': metadata,
        'num_validation_events': int(len(latencies)),
        'num_signal_events': int(hist_state['hist_pos'].sum()),
        'num_background_events': int(hist_state['hist_neg'].sum()),
        'per_sample_names': sample_names,
        'providers': providers,
        'single_output_mode': args.single_output_mode,
        'normalization': 'embedded in ONNX; raw H5 input tensors',
    }
    run['external_onnx_evaluations'] = list(external_evaluations.values())
    (output_dir / 'run_summary.json').write_text(json.dumps(run, indent=2), encoding='utf-8')

    main_row = summary_rows[0]
    print('\nFinished external ONNX evaluation:', flush=True)
    print('  model:         ', display_name, flush=True)
    print('  events:        ', len(latencies), flush=True)
    print('  signal/bg:     ', int(hist_state['hist_pos'].sum()), '/', int(hist_state['hist_neg'].sum()), flush=True)
    print('  ROC AUC:       ', main_row['roc_auc'], flush=True)
    print('  mean latency:  ', main_row['mean_event_latency_ms_onnxruntime'], 'ms/event', flush=True)
    print('  results dir:   ', output_dir, flush=True)
    print('Open the existing plotting notebook with OUTDIR pointing to this directory.', flush=True)


if __name__ == '__main__':
    main()
