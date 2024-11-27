_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc_ms.py',
    '../../../_base_/schedules/schedule.py', '../../vfa_r101_c4_JB.py',
    '../../../_base_/default_runtime.py'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        num_support_ways=21,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='VFA_JB', setting='SPLIT4_10SHOT')],
            num_novel_shots=10,
            num_base_shots=10, 
            classes='ALL_CLASSES_SPLIT4',
        )),
    val=dict(
        classes='ALL_CLASSES_SPLIT4',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/VOCdevkit/' + 'VOC2007/ImageSets/Main/test.txt'),
            dict(
                type='ann_file',
                ann_file='/home/yjhwang/workspace/FSOD/VFA1121/data/JBdevkit/test_v2.txt')
        ],
        ),
    test=dict(
        classes='ALL_CLASSES_SPLIT4'),
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/VOCdevkit/' + 'VOC2007/ImageSets/Main/test.txt'),
            dict(
                type='ann_file',
                ann_file='/home/yjhwang/workspace/FSOD/VFA1121/data/JBdevkit/test_v2.txt')
        ],
        model_init=dict(classes='ALL_CLASSES_SPLIT4'),
        unlabeled=dict(
                type='LLOSS_Unlabeled_Pool_VAE',
                ann_cfg=[
                    dict(
                        type='ann_file',
                        ann_file='/home/yjhwang/workspace/FSOD/VFA1121/data/JBdevkit/trainval_v2.txt')
                ],
                img_prefix='/home/yjhwang/workspace/FSOD/VFA1121/data/JBdevkit',
                pipeline=test_pipeline,
                test_mode=False,
                classes='ALL_CLASSES_SPLIT4'),
        )
evaluation = dict(
    interval=2000, class_splits=['BASE_CLASSES_SPLIT4', 'NOVEL_CLASSES_SPLIT4'])
checkpoint_config = dict(interval=2000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=2000)
load_from = 'checkpoint_dir/vfa_r101_c4_8xb4_voc-split1_base-training_iter_18000.pth'

# model settings
model = dict(frozen_parameters=[
    'backbone', 'shared_head',  'aggregation_layer',  'rpn_head',
])