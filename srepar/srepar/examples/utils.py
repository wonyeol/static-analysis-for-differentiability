import argparse, importlib, sys
import pyro

from srepar.lib.srepar import set_no_reparam_names

def run(run_name, cur_pckg):
    # use Pyro 1.3.0.
    assert(pyro.__version__.startswith('1.3.0'))

    # parse command-line arguments.
    train = importlib.import_module(f'.train', cur_pckg)
    parser = train.get_parser()
    args = parser.parse_args()

    # load model, guide.
    model = importlib.import_module(f'.model_{args.repar_type}', cur_pckg)
    guide = importlib.import_module(f'.guide_{args.repar_type}', cur_pckg)
    # model = importlib.import_module(f'.model', cur_pckg)
    # guide = importlib.import_module(f'.guide', cur_pckg)
    # if args.repar_type == 'score': set_no_reparam_names(True)
    # if args.repar_type == 'repar':  set_no_reparam_names([])

    # execute train.
    args.log = f"test/log/{run_name}_{args.repar_type}{args.log_suffix}.log"
    train.main(model, guide, args)
