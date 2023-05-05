import argparse
parser = argparse.ArgumentParser(description='para transfer')
#parser.add_argument('--para1', action='store_true', default=False, help='para1 -> bool type.')
parser.add_argument('--para2', type=int, default=10, help='para2 -> int type.')
parser.add_argument('--para3', type=str, default="hello", help='para3 -> str type.')
args = parser.parse_args()
print(args)
print(args.para2)
