digital_data_configs = {
	'dig-example': {
        'vid_caches_dir': '.\\datasets\\digital_paintings\\pkls_dir',  # fill this in with your dataset dir
		'pad_to_shape': (50, 50, 3),
		'im_shape': (50, 50, 3),
		'dataset': 'digital',
		'do_stretch': False,
		'crop_type': 'olap',
		'scale_factor': 0.7,
		'include_attention': False,
		'n_prev_frames': 1,
		'percent_train': 0.8,
		'normalize_frames': True,
		'frame_delta_range': (3, 90),
		'n_pred_frames': None,
    	'exclude_vids_file': None,
		'affine_breaks_file': '..\\datasets\\digital_paintings\\digital_affinebreaks.txt',  # annotations of where affine shifts happen, so we do not train on sequences that span these shifts
   		'true_starts_file': '..\\datasets\\digital_paintings\\digital_truestarts.txt',  # annotations of which videos start at the beginning of the painting
		'do_use_segment_end': True,
		'attn_params': {
			'attn_thresh': 0.05,
			'attn_erode_sigma': 3,
			'attn_dilate_sigma': 3,
		},
		'sequence_params': {
			'min_attn_area': 25,  # in pixels
			'max_attn_area': None,
			'min_good_seg_len': None,
		},
	},
}
