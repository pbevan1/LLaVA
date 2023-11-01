import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_tar', type=int, default=0)
    parser.add_argument('--end_tar', type=int, default=200)
    parser.add_argument('--dataset', type=str, default="laion-pop")
    parser.add_argument('--DEBUG', help='Runs small batch of example labels', action='store_true')

    args, _ = parser.parse_known_args()
    return args
