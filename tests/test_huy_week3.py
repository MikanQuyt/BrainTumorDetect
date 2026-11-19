"""Offline checks of split leakage, threshold selection, localization and real checkpoint resume."""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import ViTConfig, ViTModel

from data.prepare_huy_week3 import (assign_duplicate_groups, baseline_splits,
                                   extend_ds2_split, prepare_record, repair_group_split,
                                   validate_split)
from data.skull_crop import write_png
from engine.train_vit_lora_huy import (HuyDataset, evaluate_localization, fit, run_split,
                                      point_in_mask, select_threshold,
                                      summarize_localization, validate_checkpoint)
from models.vit_lora import ViTLoRA


def record(i, source="classification", group=None, label=1):
    return {"sample_id": str(i), "source": source, "sub_class_label": label,
            "has_tumor": int(label != 0), "duplicate_group": group or str(i),
            "raw_pixel_digest": group or str(i), "crop_pixel_digest": 'crop' + (group or str(i))}


class TestDataProtocol(unittest.TestCase):
    def test_y92_paths_do_not_collide_and_crop_preserves_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'dataset/navoneel/yes'
            source.mkdir(parents=True)
            image = np.zeros((64, 64, 3), np.uint8)
            image[8:56, 9:55] = 100
            mask = np.zeros((64, 64), np.uint8)
            mask[2:12, 2:12] = 255  # deliberately extends outside the detected skull crop
            write_png(source / 'mask.png', mask)
            cv2.imwrite(str(source / 'Y92.jpg'), image)
            write_png(source / 'Y92.png', image)
            a = prepare_record(root, root / 'dataset/huy_week3', 'ds1', source/'Y92.jpg', -1,
                               'classification')
            b = prepare_record(root, root / 'dataset/huy_week3', 'ds1', source/'Y92.png', -1,
                               'segmentation', mask_path=source/'mask.png')
            self.assertNotEqual(a['img_path'], b['img_path'])
            self.assertTrue((root / a['img_path']).is_file())
            exported = cv2.imread(str(root/b['mask_path']), 0)
            self.assertEqual(int((exported > 0).sum()), 100)
            self.assertEqual(b['mask_area_pixels'], 100)
            self.assertEqual(b['gt_box_orig'], [2, 2, 12, 12])

    def test_raw_and_processed_duplicate_groups_are_transitive_across_sources(self):
        records = [record(0), record(1, 'segmentation'), record(2)]
        records[1]['raw_pixel_digest'] = records[0]['raw_pixel_digest']
        records[2]['crop_pixel_digest'] = records[1]['crop_pixel_digest']
        assign_duplicate_groups(records)
        self.assertEqual(len({r['duplicate_group'] for r in records}), 1)
        records[2]['sub_class_label'] = 2
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            assign_duplicate_groups(records)

    def test_cross_partition_duplicate_is_rejected(self):
        records = [record(0, group='same'), record(1, group='same'), record(2)]
        with self.assertRaisesRegex(ValueError, 'duplicates cross'):
            validate_split(records, {'train': [0], 'val': [1], 'test': [2]})

    def test_segmentation_is_anchored_to_classification_holdouts(self):
        records = [record(0, group='train'), record(1, group='val'), record(2, group='test'),
                   record(3, 'segmentation', group='test')]
        plan = extend_ds2_split(records, {'train': [0], 'val': [1], 'test': [2]}, 3)
        self.assertEqual(plan['test'], [2, 3])
        self.assertEqual(plan['train'], [0])

    def test_baseline_sources_and_all_outer_test_folds_are_preserved(self):
        records = [record(i) for i in range(5)]
        for i, r in enumerate(records):
            r['baseline_image'] = f'yes/image{i}.jpg'
        rows = []
        for fold in range(5):
            for i, r in enumerate(records):
                part = 'test' if i == fold else ('val' if i == (fold + 1) % 5 else 'train')
                rows.append({'fold': fold + 1, 'partition': part,
                             'image': r['baseline_image'], 'duplicate_group': r['raw_pixel_digest']})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'splits.csv'
            with path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            plans = baseline_splits(records, path)
            self.assertEqual(sorted(i for s in plans for i in s['test']), list(range(5)))
            records[0]['raw_pixel_digest'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'differs from Week 2'):
                baseline_splits(records, path)

    def test_merged_group_repair_preserves_test_and_validation_priority(self):
        records = [record(0, group='test_pair'), record(1, group='test_pair'),
                   record(2, group='val_pair'), record(3, group='val_pair'), record(4)]
        original = {'fold': 1, 'train': [0, 2, 4], 'val': [3], 'test': [1]}
        repaired, audit = repair_group_split(records, original)
        self.assertEqual(repaired['train'], [4])
        self.assertEqual(repaired['val'], [2, 3])
        self.assertEqual(repaired['test'], [0, 1])
        self.assertEqual(audit['n_conflicting_groups'], 2)
        self.assertEqual(audit['n_images_moved'], 2)
        self.assertTrue(audit['all_original_test_images_preserved'])
        self.assertEqual(records[0]['week2_split'], 'train')
        validate_split(records, repaired)
        unchanged, second_audit = repair_group_split(records, repaired)
        self.assertEqual(unchanged, repaired)
        self.assertEqual(second_audit['n_images_moved'], 0)

    def test_segmentation_bridge_repairs_historical_classification_conflict(self):
        records = [record(i, source='classification') for i in range(5)]
        records.append(record(5, source='segmentation'))
        # A segmentation image joins two otherwise distinct raw classification groups.
        records[5]['raw_pixel_digest'] = records[0]['raw_pixel_digest']
        records[5]['crop_pixel_digest'] = records[1]['crop_pixel_digest']
        for i in range(5):
            records[i]['baseline_image'] = f'classification/image{i}.jpg'
        assign_duplicate_groups(records)
        original = {'fold': 1, 'train': [0, 2], 'val': [3], 'test': [1, 4]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'splits.csv'
            with path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['fold', 'partition', 'image', 'duplicate_group'])
                writer.writeheader()
                for part in ('train', 'val', 'test'):
                    for i in original[part]:
                        writer.writerow({'fold': 1, 'partition': part,
                            'image': records[i]['baseline_image'],
                            'duplicate_group': records[i]['raw_pixel_digest']})
            with self.assertRaisesRegex(ValueError, 'duplicates cross'):
                baseline_splits(records[:5], path)
            audit = []
            repaired = baseline_splits(records[:5], path, repair_merged_groups=True, repair_log=audit)[0]
            self.assertEqual(repaired['test'], [0, 1, 4])
            self.assertEqual(audit[0]['n_images_moved'], 1)
            combined = extend_ds2_split(records, repaired, 5)
            self.assertEqual(combined['test'], [0, 1, 4, 5])
            validate_split(records, combined)

    def test_raw_pixel_leakage_is_rejected_even_when_repair_is_enabled(self):
        records = [record(0, group='raw_duplicate'), record(1, group='raw_duplicate'), record(2)]
        for i, item in enumerate(records): item['baseline_image'] = f'image{i}.jpg'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'splits.csv'
            with path.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['fold', 'partition', 'image', 'duplicate_group'])
                writer.writeheader()
                for i, part in enumerate(('train', 'val', 'test')):
                    writer.writerow({'fold': 1, 'partition': part,
                        'image': records[i]['baseline_image'],
                        'duplicate_group': records[i]['raw_pixel_digest']})
            with self.assertRaisesRegex(ValueError, 'duplicates cross'):
                baseline_splits(records, path, repair_merged_groups=True)


class TestThresholdAndLocalization(unittest.TestCase):
    def test_threshold_moving_uses_tumor_probability_not_multiclass_argmax(self):
        # The last tumor is split across tumor subclasses: sum=0.39, argmax is no_tumor.
        y = [0, 0, 1, 2, 3]
        p = np.array([[.95, .02, .02, .01], [.62, .13, .13, .12],
                      [.1, .8, .05, .05], [.2, .1, .6, .1], [.61, .13, .13, .13]])
        result = select_threshold(y, p)
        self.assertTrue(result['validation_target_met'])
        self.assertAlmostEqual(result['best']['threshold'], .39)
        self.assertEqual(result['best']['recall'], 1.0)
        self.assertEqual(result['best']['specificity'], 1.0)

    def test_target_not_met_is_reported_without_extending_the_search_range(self):
        result = select_threshold([0, 1], np.array([[.9, .1], [.8, .2]]))
        self.assertFalse(result['validation_target_met'])
        self.assertEqual(result['best']['recall'], 0)
        self.assertGreaterEqual(result['best']['threshold'], .35)
        self.assertLessEqual(result['best']['threshold'], .40)

    def test_pointing_game_and_detector_misses_remain_in_denominator(self):
        mask = np.zeros((10, 10), np.uint8)
        mask[2:5, 2:5] = 1
        self.assertTrue(point_in_mask([3, 3], mask))
        self.assertFalse(point_in_mask([-3, 3], mask))
        rows = [{'detected': True, 'box_iou': 1., 'point_hit': True},
                {'detected': False, 'box_iou': 1., 'point_hit': True}]
        result = summarize_localization(rows)
        self.assertEqual(result['mean_box_iou'], 1.)
        self.assertEqual(result['end_to_end_mean_box_iou'], .5)
        self.assertEqual(result['end_to_end_pointing_game_accuracy'], .5)


class TestTrainingResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_fixture(self, root, multiclass=False):
        records = []
        for i in range(8):
            image = np.zeros((64, 64, 3), np.uint8)
            image[6:58, 6:58] = 60 + i * 15
            image[20:35, 20:35] = 220
            mask = np.zeros((64, 64), np.uint8)
            mask[20:35, 20:35] = 255
            img_path, mask_path = root/f'{i}.png', root/f'{i}_mask.png'
            write_png(img_path, image); write_png(mask_path, mask)
            label = i % (4 if multiclass else 2)
            records.append({**record(i, label=label), 'dataset': 'indk214' if multiclass else 'navoneel',
                            'img_path': img_path.name, 'mask_path': mask_path.name if multiclass and label else None,
                            'has_mask': bool(multiclass and label), 'gt_box': [20, 20, 35, 35] if label and multiclass else None})
        return records

    def model(self, multiclass=False):
        torch.manual_seed(17)
        backbone = ViTModel(ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                     intermediate_size=64, image_size=224, patch_size=16),
                            add_pooling_layer=False)
        return ViTLoRA(backbone=backbone, pretrained=False, num_classes=4 if multiclass else 2,
                       lora_r=8, lora_alpha=16, sfp_dim=16, use_box_head=multiclass)

    def args(self, epochs=2, multiclass=False):
        return argparse.Namespace(epochs=epochs, resume='auto', device='cpu', amp=False,
                                  batch_size=2, workers=0, seed=42, patience=5,
                                  lr_patience=1, min_delta=1e-4, lr_lora=1e-3,
                                  lr_head=1e-3, lr_box=1e-3, weight_decay=.01,
                                  loss='focal', label_smoothing=.05, box_loss_weight=1.,
                                  task='multiclass' if multiclass else 'binary')

    def test_resume_matches_uninterrupted_weights_optimizer_and_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_fixture(root)
            train, val = HuyDataset(records[:4], root, train=True), HuyDataset(records[4:], root)
            identity = {'fixture': 'resume'}
            full = self.model()
            fit(full, train, val, self.args(2), root/'full', identity)
            interrupted = self.model()
            fit(interrupted, train, val, self.args(1), root/'resumed', identity)
            resumed = self.model()
            fit(resumed, train, val, self.args(2), root/'resumed', identity)
            a = torch.load(root/'full/loss_latest.pth', weights_only=False)
            b = torch.load(root/'resumed/loss_latest.pth', weights_only=False)
            self.assertEqual(a['epoch'], b['epoch'])
            self.assertEqual(a['history'], b['history'])
            self.assertEqual(a['scheduler'], b['scheduler'])
            self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], rtol=0, atol=0)
            for key, state in a['optimizer']['state'].items():
                for name, value in state.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=0, atol=0)
            self.assertTrue((root/'resumed/loss_best.pth').is_file())
            with self.assertRaisesRegex(ValueError, 'config/data/code changed'):
                validate_checkpoint(b, {'fixture': 'different'}, b['frozen_sha256'])
            with self.assertRaisesRegex(ValueError, 'Frozen pretrained backbone differs'):
                validate_checkpoint(b, identity, 'wrong-base')

    def test_real_ds2_training_and_auto_prompt_evaluation_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_fixture(root, multiclass=True)
            model = self.model(multiclass=True)
            args = self.args(1, multiclass=True)
            result = fit(model, HuyDataset(records[:4], root), HuyDataset(records[4:], root),
                         args, root/'run', {'fixture': 'box-smoke'})
            self.assertEqual(result['epochs_completed'], 1)
            evaluation = evaluate_localization(model, HuyDataset([records[1]], root), args,
                                               .35, root/'run')
            self.assertEqual(evaluation['overall']['n'], 1)
            self.assertGreaterEqual(evaluation['overall']['mean_box_iou'], 0.)
            self.assertTrue((root/'run/localization_predictions.csv').is_file())

    def test_validation_early_stopping_survives_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_fixture(root)
            args = self.args(epochs=20)
            args.patience = 1
            args.min_delta = 100.0  # only the first finite validation loss is significant
            result = fit(self.model(), HuyDataset(records[:4], root), HuyDataset(records[4:], root),
                         args, root/'run', {'fixture': 'stopping'})
            self.assertTrue(result['early_stopped'])
            self.assertEqual(result['epochs_completed'], 2)
            resumed = fit(self.model(), HuyDataset(records[:4], root), HuyDataset(records[4:], root),
                          args, root/'run', {'fixture': 'stopping'})
            self.assertEqual(resumed['epochs_completed'], 2)

    def test_complete_run_writes_thresholds_metrics_and_uses_completed_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self.make_fixture(root)
            split = {'fold': 1, 'train': [0, 1, 2, 3], 'val': [4, 5], 'test': [6, 7]}
            args = self.args(epochs=1)
            args.out, args.root = root/'results', root
            args.model_name, args.offline = 'test-backbone', True

            def factory(**_kwargs):
                torch.manual_seed(17)
                backbone = ViTModel(ViTConfig(hidden_size=32, num_hidden_layers=12,
                    num_attention_heads=4, intermediate_size=64, image_size=224, patch_size=16),
                    add_pooling_layer=False)
                return ViTLoRA(backbone=backbone, pretrained=False, num_classes=2,
                               lora_r=8, lora_alpha=16, use_box_head=False)

            with patch('engine.train_vit_lora_huy.ViTLoRA', side_effect=factory) as constructor:
                result = run_split(args, 'ds1', records, split)
                again = run_split(args, 'ds1', records, split)
                self.assertEqual(constructor.call_count, 1)
            self.assertEqual(result, again)
            out = args.out/'ds1/fold_01'
            for filename in ('loss_best.pth', 'loss_latest.pth', 'test_metrics.json',
                             'threshold_selection.json', 'validation_threshold_curve.csv',
                             'test_predictions.csv', 'learning_curves.png'):
                self.assertTrue((out/filename).is_file(), filename)
            self.assertEqual(result['threshold_selection']['selected_on'],
                             'validation classification samples only')


if __name__ == '__main__':
    unittest.main()
