import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))
from scripts.audit_eeg_sessions import audit, signal_statistics
from scripts.run_eeg_session_category import metrics, run


def test_alpha_signal_and_nonfinite():
    signal = np.tile(np.sin(2*np.pi*10*np.arange(800)/200), (62, 1))
    result = signal_statistics(signal, 200)
    assert result["band_relative_power"]["alpha"] > 0.99
    signal[0, 0] = np.nan
    assert signal_statistics(signal, 200)["nonfinite_samples"] == 1


def test_npz_order_semantics_and_independent_alignment(tmp_path):
    rows, events = [], []
    for number in (1, 2, 3):
        session = f"session{number}"
        path, metadata = tmp_path / f"{session}.npz", tmp_path / f"{session}.csv"
        np.savez(path, eeg=np.random.default_rng(number).normal(size=(2,62,800)).astype('float32'),
                 filename=np.array(['01-001','01-002']), length=np.array([800,800]),
                 mask=np.ones((2,800),bool), sfreq=200., order_index=np.array([1,2]),
                 playback_order_index=np.array([20,10]), channel_names=np.array([f'C{i}' for i in range(62)]))
        meta_rows = []
        for i in range(2):
            video = f'01-{i+1:03d}'
            onset = 4000 - 2000*i
            meta_rows.append(dict(video_id=video, sfreq=200, n_times=800, duration_sec=4,
                                  n_eeg_channels=62, sorted_index=i+1, order_index=20-10*i,
                                  playback_order_index=20-10*i, onset_sample=onset,
                                  annotation_event_sample=onset+10))
            rows.append(dict(video_id=video, session=session, npz_path=str(path), trial_index=str(i),
                             length_samples='800', metadata_path=str(metadata), metadata_row=str(i),
                             sfreq='200', duration_sec='4'))
            events.append(dict(session=session, video_id=video, onset_sample=onset))
        with metadata.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(meta_rows[0]))
            writer.writeheader()
            writer.writerows(meta_rows)
    result = audit(rows)
    assert result['internal_mapping_status'] == 'PASS'
    assert result['independent_stimulus_alignment'] == 'NOT_VERIFIED'
    assert result['signal_summary']['session2']['onset_minus_annotation_sample_counts'] == {-10:2}
    assert audit(rows, events)['independent_stimulus_alignment'] == 'MATCHED_PROVIDED_EVENTS'
    events[0]['video_id']='02-001'
    assert audit(rows, events)['internal_mapping_status'] == 'FAIL'
    rows[0]['trial_index']='1'
    assert any('filename disagrees' in e for e in audit(rows)['errors'])


def test_category_controls_cpu(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    package = dict(eeg=torch.randn(24,3,62,128), categories=torch.arange(24)%6,
                   normalization_mean=torch.zeros(3,62), normalization_std=torch.ones(3,62),
                   splits={'train':list(range(12)),'validation':list(range(12,18)),'test':list(range(18,24))},
                   ids=[str(i) for i in range(24)])
    prepared=tmp_path/'prepared.pt'
    torch.save(package,prepared)
    args=Namespace(prepared=prepared,output_root=tmp_path/'runs',protocol='holdout_s3',
                   model='compact',seed=42,epochs=1,batch_size=6,device='cpu')
    for model in ('compact','ridge'):
        args.model=model
        run(args)
        path=args.output_root/'holdout_s3'/model/'seed42/report.json'
        report=json.loads(path.read_text())
        assert report['reports']['train/session3']['heldout_session']
        assert report['reports']['test/session3']['video_count']==6
        assert report['protocol']['normalization_fit_sessions']==[0,1]
    assert metrics(torch.eye(6),torch.arange(6))['accuracy']==1.
