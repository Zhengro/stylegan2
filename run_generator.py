# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import time

import pretrained_networks

#----------------------------------------------------------------------------


def generate_images(network_pkl, seeds, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx+1, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        start = time.time()
        images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        print('Inference Time: {:.2f}s'.format(time.time() - start))
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))

#----------------------------------------------------------------------------


def generate_ws(network_pkl, seeds, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    w_avg = Gs.get_var('dlatent_avg')  # [component]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    Ws = np.zeros((len(seeds), 18, 512))

    for seed_idx, seed in enumerate(seeds):
        print('Generating W for seed %d (%d/%d) ...' % (seed, seed_idx+1, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [1, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        start = time.time()
        w = Gs.components.mapping.run(z, None)  # [1, layer, component]
        w = w_avg + (w - w_avg) * truncation_psi  # [1, layer, component]
        print('Time: {}s'.format(time.time() - start))
        Ws[seed_idx, :, :] = w

    np.save('W_{}.npy'.format(len(seeds)), Ws)

#----------------------------------------------------------------------------


def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('col_styles: {}'.format(col_styles))
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg')  # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds])  # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None)  # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi  # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}  # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs)  # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------


def style_mixing_gradual_change(network_pkl, row_seeds, col_seeds, truncation_psi, minibatch_size=4):
    col_styles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 17]
    print('col_styles: {}'.format(col_styles))
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg')  # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds])  # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None)  # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi  # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}  # [layer, component]
    image_dict = {}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            for col_style in col_styles:
                w[0:col_style+1] = w_dict[col_seed][0:col_style+1]
                image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]  # [height, width, channel]
                image_dict[(row_seed, col_seed, col_style)] = image

    print('Saving images...')
    for (row_seed, col_seed, col_style), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d 0-%d.png' % (row_seed, col_seed, col_style)))

#----------------------------------------------------------------------------


def style_mixing_noise(network_pkl, row_seeds, col_seeds, col_styles, truncation_psi, randomize_noise, minibatch_size=4):
    print('col_styles: {}'.format(col_styles))
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg')  # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = randomize_noise
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds])  # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None)  # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi  # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}  # [layer, component]
    image_dict = {}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            for i_noise in range(10):
                image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]  # [height, width, channel]
                image_dict[(row_seed, col_seed, i_noise)] = image

    print('Saving images...')
    for (row_seed, col_seed, i_noise), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d noise%d.png' % (row_seed, col_seed, i_noise)))

#----------------------------------------------------------------------------


def style_mixing_multiple(network_pkl, row_seeds, col_seeds, truncation_psi, minibatch_size=4):
    col_styles = [[6, 7], [0, 3]]
    print('col_styles: {}'.format(col_styles))
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg')  # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds])  # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None)  # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi  # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}  # [layer, component]
    image_dict = {}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        w = w_dict[row_seed].copy()
        for col_style, col_seed in zip(col_styles, col_seeds):
            w[col_style] = w_dict[col_seed][col_style]
        image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]  # [height, width, channel]
        image_dict[(row_seed, col_seeds[0], col_seeds[1])] = image

    print('Saving images...')
    for (row_seed, col_seeds[0], col_seeds[1]), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d-%d.png' % (row_seed, col_seeds[0], col_seeds[1])))

#----------------------------------------------------------------------------


def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------


_examples = '''examples:
    
    # Example of image generation
    python run_generator.py generate-images --network=stylegan2-ffhq-config-f.pkl --seeds=0-9 --truncation-psi=1.0
    
    # Example of style mixing
    python run_generator.py style-mixing-example --network=stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0

    # Example of gradual change
    python run_generator.py style-mixing-gradual-change --network=stylegan2-ffhq-config-f.pkl --row-seeds=100 --col-seeds=85 --truncation-psi=1.0

    # Example of noise effect
    python run_generator.py style-mixing-noise --network=stylegan2-ffhq-config-f.pkl --row-seeds=100 --col-seeds=85 --col-styles=0-1 --randomize-noise True --truncation-psi=1.0
    
    # Example of style mixing of three references
    python run_generator.py style-mixing-multiple --network=stylegan2-ffhq-config-f.pkl --row-seeds=100 --col-seeds=85,20 --truncation-psi=1.0
'''

#----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_generate_ws = subparsers.add_parser('generate-ws', help='Generate Ws')
    parser_generate_ws.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_ws.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_ws.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_ws.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_gradual_change = subparsers.add_parser('style-mixing-gradual-change', help='Generate gradual change')
    parser_style_mixing_gradual_change.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_gradual_change.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_gradual_change.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_gradual_change.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_gradual_change.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_noise = subparsers.add_parser('style-mixing-noise', help='Pose noises')
    parser_style_mixing_noise.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_noise.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_noise.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_noise.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_noise.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_noise.add_argument('--randomize-noise', type=bool, help='Randomize noise (default: %(default)s)', default=False)
    parser_style_mixing_noise.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_multiple = subparsers.add_parser('style-mixing-multiple', help='Generate from multiple references')
    parser_style_mixing_multiple.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_multiple.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_multiple.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_multiple.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_multiple.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator.generate_images',
        'generate-ws': 'run_generator.generate_ws',
        'style-mixing-example': 'run_generator.style_mixing_example',
        'style-mixing-gradual-change': 'run_generator.style_mixing_gradual_change',
        'style-mixing-noise': 'run_generator.style_mixing_noise',
        'style-mixing-multiple': 'run_generator.style_mixing_multiple'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
