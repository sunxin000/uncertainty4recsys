from scipy import stats
import numpy as np
import argparse 


def calculate_confidence_interval(array):
    mean = np.mean(array, axis=-1)
    print(mean)
    std = np.std(array, axis=-1)
    # std = np.std(array, axis=-1,  ddof=1)
    # interval = stats.t.interval(alpha=0.95, df=len(array)-1, loc=mean, scale=std, )

    interval = stats.norm.interval(alpha=0.95, loc=mean, scale=std)
    return mean , mean - interval[0]
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--array', type=float,  nargs='+')
    parser.add_argument('--name')
    args = parser.parse_args()


    result = np.loadtxt(f'results/{args.name}', dtype=float, delimiter=' ', usecols=range(6))
    array = result.T

    # array = np.asarray(args.array)

    assert array.shape[1] == 10

    mean, diff = calculate_confidence_interval(array)
    # rst = np.vstack((mean, diff))
    rst = [f'{m:.4f}±{d:.4f}' for m, d in zip(mean, diff)]
    # np.savetxt(f'results/mean_{args.name}', rst, fmt='%.4f')
    with open(f'results/mean_{args.name}', 'w') as file:
        file.write(' '.join(rst))


