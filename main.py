from gauging_proximity import Perception
import numpy as np
import os
from clt_tools import load_data, mkdir_p, draw_clusters
import cProfile
import pstats


def prepare_test_cases():
    test_cases = {
        # '2d-10c': 'data/2d-10c.txt',
        # '2d-4c-1': 'data/2d-4c-1.txt',
        # '2d-4c-3': 'data/2d-4c-3.txt',
        # '2sp2glob': 'data/2sp2glob.txt',
        '3-spiral': 'data/3-spiral.txt',
        # '3MC': 'data/3MC.txt',
        '3_blobs': 'data/3_blobs.txt',
        # 'aml28': 'data/aml28.txt',
        # 'atom': 'data/atom.txt',
        # 'chainlink': 'data/chainlink.txt',
        # 'complex8': 'data/complex8.txt',
        # 'complex9': 'data/complex9.txt',
        'compound': 'data/compound.txt',
        # 'cuboids': 'data/cuboids.txt',
        # 'cure-t0-2000n-2D': 'data/cure-t0-2000n-2D.txt',
        # 'curves1': 'data/curves1.txt',
        # 'curves2': 'data/curves2.txt',
        'D31_small': 'data/D31_small.txt',
        # 'dartboard1': 'data/dartboard1.txt',
        'dartboard2': 'data/dartboard2.txt',
        'disk-4000n': 'data/disk-4000n.txt',
        # 'donut1': 'data/donut1.txt',
        # 'donut2': 'data/donut2.txt',
        # 'donut3': 'data/donut3.txt',
        # 'donutcurves': 'data/donutcurves.txt',
        # 'ds2c2sc13': 'data/ds2c2sc13.txt',
        # 'fourty': 'data/fourty.txt',
        # 'g2_2_10': 'data/g2_2_10.txt',
        # 'gaussians1': 'data/gaussians1.txt',
        # 'hepta': 'data/hepta.txt',
        # 'hypercube': 'data/hypercube.txt',
        # 'jain': 'data/jain.txt',
        # 'long1': 'data/long1.txt',
        # 'long2': 'data/long2.txt',
        # 'long3': 'data/long3.txt',
        # 'lsun': 'data/lsun.txt',
        # 'pmf': 'data/pmf.txt',
        # 'shapes': 'data/shapes.txt',
        # 'smile1': 'data/smile1.txt',
        # 'smile2': 'data/smile2.txt',
        # 'smile3': 'data/smile3.txt',
        'spherical_4_3': 'data/spherical_4_3.txt',
        # 'spherical_6_2': 'data/spherical_6_2.txt',
        # 'target': 'data/target.txt',
        # 'triangle1': 'data/triangle1.txt',
        # 'twenty': 'data/twenty.txt',
        # 'wingnut': 'data/wingnut.txt',
        # 'zelnik1': 'data/zelnik1.txt',
        # 'zelnik3': 'data/zelnik3.txt',
        # 'zelnik4': 'data/zelnik4.txt',
        # 'zelnik5': 'data/zelnik5.txt',
        # 'zelnik6': 'data/zelnik6.txt',
        # 'banana': 'data/banana.txt',
        # 'cassini': 'data/cassini.txt',
    # not well separated below
    #     '2d-20c': 'data/2d-20c.txt',
        '2d-3c': 'data/2d-3c.txt',
    #     '2d-4c-2': 'data/2d-4c-2.txt',
    #     '2dnormals': 'data/2dnormals.txt',
        'aggregation': 'data/aggregation.txt',
    #     'impossible': 'data/impossible.txt',
    #     'pathbased': 'data/pathbased.txt',
        'blobs': 'data/blobs.txt',
        'cure-t1-2000n-2D': 'data/cure-t1-2000n-2D.txt',
        'cure-t2-4k': 'data/cure-t2-4k.txt',
    #     'D31': 'data/D31.txt',
    #     'dense-disk-3000': 'data/dense-disk-3000.txt',
    #     'dense-disk-5000': 'data/dense-disk-5000.txt',
        'diamond9': 'data/diamond9.txt',
    #     'disk-1000n': 'data/disk-1000n.txt',
    #     'disk-3000n': 'data/disk-3000n.txt',
    #     'disk-4500n': 'data/disk-4500n.txt',
    #     'disk-4600n': 'data/disk-4600n.txt',
    #     'disk-5000n': 'data/disk-5000n.txt',
    #     'dpb': 'data/dpb.txt',
    #     'dpc': 'data/dpc.txt',
    #     'DS-577': 'data/DS-577.txt',
    #     'DS-850': 'data/DS-850.txt',
    #     'ds3c3sc6': 'data/ds3c3sc6.txt',
    #     'ds4c2sc8': 'data/ds4c2sc8.txt',
    #     'elliptical_10_2': 'data/elliptical_10_2.txt',
    #     'elly-2d10c13s': 'data/elly-2d10c13s.txt',
    #     'flame': 'data/flame.txt',
    #     'engytime': 'data/engytime.txt',
    #     'longsquare': 'data/longsquare.txt',
    #     'R15': 'data/R15.txt',
    #     'rings': 'data/rings.txt',
    #     'simplex': 'data/simplex.txt',
        'sizes1': 'data/sizes1.txt',
    #     'sizes2': 'data/sizes2.txt',
    #     'sizes3': 'data/sizes3.txt',
    #     'sizes4': 'data/sizes4.txt',
    #     'sizes5': 'data/sizes5.txt',
    #     'spherical_5_2': 'data/spherical_5_2.txt',
    #     'spiralsquare': 'data/spiralsquare.txt',
    #     'square1': 'data/square1.txt',
    #     'square2': 'data/square2.txt',
    #     'square3': 'data/square3.txt',
    #     'square4': 'data/square4.txt',
    #     'st900': 'data/st900.txt',
    #     'tetra': 'data/tetra.txt',
    #     'threenorm': 'data/threenorm.txt',
    #     'triangle2': 'data/triangle2.txt',
        'twodiamonds': 'data/twodiamonds.txt',
    #     'xclara': 'data/xclara.txt',
    #     'xor': 'data/xor.txt',
    #     'zelnik2': 'data/zelnik2.txt',
        # 'glass': 'data/glass.arff',
        # 'arrhythmia': 'data/arrhythmia.arff',
        # 'balance-scale': 'data/balance-scale.arff',
        # 'cpu': 'data/cpu.arff',
        # 'dermatology': 'data/dermatology.arff',
        # 'ecoli': 'data/ecoli.arff',
        # 'haberman': 'data/haberman.arff',
        # 'heart-statlog': 'data/heart-statlog.arff',
        # 'iono': 'data/iono.arff',
        # 'iris': 'data/iris.arff',
        # 'segment': 'data/segment.arff',
        # 'sonar': 'data/sonar.arff',
        # 'tae': 'data/tae.arff',
        # 'thy': 'data/thy.arff',
        # 'vehicle': 'data/vehicle.arff',
        # 'vowel': 'data/vowel.arff',
        # 'wdbc': 'data/wdbc.arff',
        # 'wine': 'data/wine.arff',
        # 'wisc': 'data/wisc.arff',
        # 'yeast': 'data/yeast.arff',
        # 'zoo': 'data/zoo.arff',
        # 'USPS': 'USPS/usps_hog_features.txt',
    }
    return test_cases


def evaluation(cases_data):
    evaluation_root = 'plots/'
    version = '20250403_whole-algorithm'

    for key, case in cases_data.items():
        # log_path = evaluation_root + version + '/' + key
        log_path = evaluation_root + version
        mkdir_p(log_path)
        # define the model
        print(f"Processing {key}...")
        X = load_data(case, key)[0]
        model = Perception(log_path=log_path)

        # fit the model

        yhat, xhat = model.fit(X)

        # save the labels in text file
        folder_name = f"labels/{version}/"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        file_path = os.path.join(folder_name, f'{key}.txt')
        np.savetxt(file_path, yhat, fmt='%d')

        dimension = len(X[0])
        if dimension == 2 or dimension == 3:
            f = draw_clusters(xhat, yhat)
            f.savefig(f'{log_path}/{key}.png', bbox_inches='tight')
            f.clf()


if __name__ == '__main__':
    evaluation(prepare_test_cases())
    # cProfile.run('evaluation(prepare_test_cases())', 'profiling_results')
    # # Print profiling results
    # with open('profiling_summary.txt', 'w') as f:
    #     p = pstats.Stats('profiling_results', stream=f)
    #     p.strip_dirs().sort_stats('cumulative').print_stats(20)
    #
    # # Read and print the results
    # with open('profiling_summary.txt', 'r') as f:
    #     print(f.read())
