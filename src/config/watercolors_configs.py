watercolors_data_configs = {
	'watercolor-example': {
		'pad_to_shape': (50, 50, 3),
		'im_shape': (50, 50, 3),
		'dataset': 'watercolors_batch123_pruned',
        'vid_caches_dir': '.\\datasets\\watercolors\\vid_caches_shifted',  # fill this in with your dataset dir
		'crop_type': 'olap',
		'do_stretch': False,
		'scale_factor': 0.7,
		'include_attention': False,
		'n_prev_frames': 1,
		'percent_train': 0.8,
		'normalize_frames': True,
		'frame_delta_range': (200, 1000),
		'n_pred_frames': None,
    	'exclude_vids_file': None,
		'affine_breaks_file': None,
        'true_starts_file': None,
        'frame_shifts_dir': None,
        'do_exclude_bad_attn_from_dir': None,
		'do_use_segment_end': False,
		'attn_params': {
			'attn_thresh': 0.05,
			'attn_erode_sigma': 3,
			'attn_dilate_sigma': 3,
		},
		'sequence_params': {
			'min_attn_area': 25,  # in pixels
			'max_attn_area': 0.8,
			'min_good_seg_len': None,
		},
	}
}
