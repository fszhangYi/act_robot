import matplotlib
matplotlib.use('Agg')

import json
import matplotlib.pyplot as plt
import argparse

def plot_loss(json_path, y_min=None, y_max=None, output='loss_curve.png'):

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(list(data.keys()))

    losses = [item['loss'] for item in data['train']]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(losses, linewidth=1.5)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)

    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figure has been saved {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw loss curve")
    parser.add_argument('--json', default='train_history.json', help="Json file path")
    parser.add_argument('--y-min', type=float, help="Y min")
    parser.add_argument('--y-max', type=float, help="Y max")
    parser.add_argument('--output', default='loss_curve.png', help="Figure output path")
    args = parser.parse_args()

    plot_loss(args.json, args.y_min, args.y_max, args.output)
