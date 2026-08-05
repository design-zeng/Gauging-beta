import numpy as np
import math
import matplotlib.pyplot as plt
import time
import matplotlib
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from local_density_calculation import lower_density_selection


START_PLOT = 0
CONTINUATION_DEBUG = True
PROXIMITY_SHOW = True
PROXIMITY_DEBUG = True
SAVE_FIG = False
LOCALITY = True
WITHLINE = True
if SAVE_FIG:
    matplotlib.use("Agg")


class Perception:

    def __init__(self, log_path=None, metric='euclidean', linkage='single'):
        # metric can be “euclidean” or "precomputed".
        # If metric is “precomputed”, a distance matrix is needed as input for the fit method.
        # linkage can be “average” or “single”.
        self.history_info = {
            'inter_intra_dist_ratio': [],
            'point_base_dist_Ratio': [],
            'statics': []
        }
        self.continuation_data = {
            'red std': [],
            'green std': [],
            'red N': [],
            'green N': [],
            'mean std': [],
            'std std': [],
            'std ratio': [],
            'threshold': [],
            'continuation': [],
            'merged': [],
        }
        if log_path is not None:
            self.log_path = log_path
        else:
            self.log_path = 'plots'

        self.THRESHOLD_CONTINUATION = 0.15
        self.NUM_NEAREST_CLUSTERS = 5
        self.metric = metric
        self.linkage = linkage

    def fit(self, X, distance_matrix=None):
        self.X = X
        # self.distance_matrix is the distance matrix of the points
        self.distance_matrix = distance_matrix
        self.DIMENSION = len(X[0])
        self.X_RANGE = min(X[:, 0]), max(X[:, 0])
        self.Y_RANGE = min(X[:, 1]), max(X[:, 1])
        # self.DIST_MATRIX is for clusters
        self.DIST_MATRIX = np.full((len(X), len(X)), np.inf)
        self.FAIL_MATRIX = np.empty((len(X), len(X), 2))

        # 'data' is the list of indices of the points in the cluster
        self.initial_clusters = {
            id: {
                'label': id,
                'data': [id],
                'center': x,
                'std_dist': 0,
                'past_dists': [],
                'merging_dists': [],
                'past_densities': [],
                'past_std': [],
                'traces': []
            }
            for id, x in enumerate(self.X)}

        start_time = time.time()
        # in the following line of code, self.DIST_MATRIX is also initialized.
        # What is the self.points_dist?
        self.clusters_dist, self.points_dist = self.initiate_dists()
        early_stop = False

        # Identify border points from the dataset
        self.border_pts = lower_density_selection(X).run()

        while len(self.clusters_dist):
            i = 0
            # sorted_clusters_dist = sorted(self.clusters_dist.items(), key=lambda item: item[1])
            pre_length = len(self.initial_clusters)

            k_nearest_clusters = self.get_indices_of_k_smallest(k=X.shape[0], matrix=self.DIST_MATRIX, sorted=True)
            # self.GRAPH = self.construct_graph(k_nearest_clusters)
            # k_nearest_clusters = self.get_indices_of_k_smallest(k=(pre_length * pre_length) // 4, matrix=self.DIST_MATRIX, sorted=True)
            last_merged_cluster = None

            while i < len(k_nearest_clusters[0]):
                c1, c2 = k_nearest_clusters[:, i]
                print(f"Round: {i} -- {c1} and {c2}")

                # if self.DIST_MATRIX[c1][c2] > 100 * self.MIN_BTN_CLUSTER_DIST:
                #     break

                # if c1 in self.initial_clusters.keys() and c2 in self.initial_clusters.keys():
                if ~np.isinf(self.DIST_MATRIX[c1][c2]):
                    # to avoid replicated merging of two failed clusters.
                    # n_clusters = len(self.initial_clusters[c1]['data']) + len(self.initial_clusters[c2]['data'])
                    # if self.FAIL_MATRIX[c1][c2][0] == n_clusters and self.FAIL_MATRIX[c1][c2][1]:
                    #     i += 1
                    #     continue

                    if last_merged_cluster is not None:
                        _c = self.get_indices_of_k_smallest(1, self.DIST_MATRIX[last_merged_cluster, :])[:, 0][0]

                        if self.DIST_MATRIX[last_merged_cluster][_c] < self.DIST_MATRIX[c1][c2]:
                            c1, c2 = last_merged_cluster, _c
                        else:
                            i += 1
                        last_merged_cluster = None
                    else:
                        i += 1

                    # border points not doing clustering
                    if c1 in self.border_pts or c2 in self.border_pts:
                        i += 1
                        continue

                    is_complete, merged_cluster = self.merge_clusters(c1, c2)
                    if is_complete:
                        last_merged_cluster = merged_cluster
                    #     if len(self.initial_clusters) < START_PLOT:
                    #         self.plot(WITHLINE)
                    #         if self.DIMENSION > 3:
                    #             self.save_clusters()
                else:
                    i += 1

            print(f"The number of existing clusters: {len(self.initial_clusters)}")
            if pre_length == len(self.initial_clusters) or early_stop:
                break
                res = np.column_stack((self.X, Y_hat))
                np.savetxt(f'/k.out', res)

                # self.plot(withline=True)

        self.assign_border_points()

        print(f"Clustering finished in {(time.time() - start_time) / 60} minutes")
        Y_hat = np.empty(len(self.X))
        for k, v in self.initial_clusters.items():
            for x in v['data']:
                Y_hat[x] = int(v['label'])
        return Y_hat, self.X


    def assign_border_points(self):
        to_assign = {}
        existing_clusters = list(set(self.initial_clusters.keys()) - set(self.border_pts))
        for c in existing_clusters:
            to_assign[c] = {}  # to_assign[c] = {border_point: distance}
        # for each border point, find the nearest cluster (non-border point)
        for p in self.border_pts:
            nearest_cluster = self.find_closet_cluster(p, clusters_range=existing_clusters)
            to_assign[nearest_cluster][p] = self.clusters_dist[frozenset((p, nearest_cluster))]['distance']
        # for each existing cluster, merge the border point according to their distance
        for c, border_points in to_assign.items():
            sorted_border_points = sorted(border_points.items(), key=lambda x: x[1])
            for p, d in sorted_border_points:
                mergeable, _, _ = self.vision_generic(c, p)
                if mergeable:
                    self.merge_2_cluster(c, p)


    def construct_graph(self, cluster_pairs):
        existing_clusters = []
        pairs = []
        for i in range(len(cluster_pairs[0])):
            if cluster_pairs[0][i] not in existing_clusters or cluster_pairs[1][i] not in existing_clusters:
                pairs.append([cluster_pairs[0][i], cluster_pairs[1][i],
                              self.DIST_MATRIX[cluster_pairs[0][i]][cluster_pairs[1][i]]])
                if cluster_pairs[0][i] not in existing_clusters:
                    existing_clusters.append(cluster_pairs[0][i])
                if cluster_pairs[1][i] not in existing_clusters:
                    existing_clusters.append(cluster_pairs[1][i])
        return np.array(pairs)

    def initiate_dists(self):
        clusters_dists = {}
        points_dists = {}

        min_dist = np.inf
        sum_dist = 0
        count = 0
        DISTS = []

        list_clusters = list(self.initial_clusters.keys())
        for i in range(len(list_clusters)):
            for j in range(i + 1, len(list_clusters)):
                _d = self.compute_2_clusters_dist(list_clusters[i], list_clusters[j])
                # 'distance_info' is a dictionary containing all the information about the distance between two clusters.
                clusters_dists[frozenset([list_clusters[i],
                                 list_clusters[j]])] = _d
                # if self.linkage == 'average':
                #     _d = _d['average_dist']['distance']
                # elif self.linkage == 'single':
                #     _d = _d['near_dist']['distance']
                _d = _d['distance']
                # _d is a number in the following
                self.DIST_MATRIX[j][i], self.DIST_MATRIX[i][j] = _d, _d

                if list_clusters[i] in points_dists.keys():
                    points_dists[list_clusters[i]].append([list_clusters[j], _d])
                else:
                    points_dists[list_clusters[i]] = [[list_clusters[j], _d]]
                if list_clusters[j] in points_dists.keys():
                    points_dists[list_clusters[j]].append([list_clusters[i], _d])
                else:
                    points_dists[list_clusters[j]] = [[list_clusters[i], _d]]

                DISTS.append(_d)

        DISTS = sorted(DISTS)
        # to avoid extremly small min inter-clusters distance
        # for i in range(1, len(DISTS)):
        #     if DISTS[i] / DISTS[i-1] < 1.5:
        #         self.MIN_BTN_CLUSTER_DIST = DISTS[i]
        #         break
        self.MIN_BTN_CLUSTER_DIST = DISTS[int(len(DISTS)/100)]
        # self.MEAN_BTN_CLUSTER_DIST = np.mean(DISTS)
        self.MEAN_WTN_CLUSTER_DIST = 0.0001

        for k, v in points_dists.items():
            points_dists[k] = list(sorted(v, key=lambda x: x[1]))

        return clusters_dists, points_dists

    def compute_2_clusters_dist(self, c1, c2):
        '''
        calculate the distance between two clusters, c1 and c2.
        :param c1: First cluster
        :param c2: Second cluster
        :return: A dictionary with near_dist, center_dist, and average_dist
        '''
        closest_points = self.closest_points_btn_2_clusters(c1, c2)

        if self.metric == 'euclidean':
            if self.linkage == 'single':
                # Calculate near_dist
                near_dist = np.linalg.norm(self.X[closest_points[0]] - self.X[closest_points[1]]), closest_points[0], \
                            closest_points[1]
                dist = {'distance': near_dist[0], 'reference_points': {c1: near_dist[1], c2: near_dist[2]}}
            elif self.linkage == 'average':
                # **Optimized average_dist using scipy's cdist**
                average_dist = np.mean(
                    cdist(self.X[self.initial_clusters[c1]['data']], self.X[self.initial_clusters[c2]['data']]))
                dist = {'distance': average_dist, 'reference_points': {c1: closest_points[0], c2: closest_points[1]}}

            # Calculate center_dist
            # center_dist = np.linalg.norm(self.initial_clusters[c1]['center'] - self.initial_clusters[c2]['center']), \
            #               self.initial_clusters[c1]['center'], self.initial_clusters[c2]['center']

        elif self.metric == 'precomputed':
            if self.linkage == 'single':
                # Directly retrieve distances from the precomputed distance matrix
                near_dist = self.distance_matrix[closest_points[0], closest_points[1]], closest_points[0], closest_points[1]
                dist = {'distance': near_dist[0], 'reference_points': {c1: near_dist[1], c2: near_dist[2]}}
            # center_dist = self.distance_matrix[c1, c2], c1, c2
            elif self.linkage == 'average':
                # Calculate average_dist from the precomputed distance matrix
                data_c1 = self.initial_clusters[c1]['data']
                data_c2 = self.initial_clusters[c2]['data']
                average_dist = np.mean([self.distance_matrix[i, j] for i in data_c1 for j in data_c2])
                dist = {'distance': average_dist, 'reference_points': {c1: closest_points[0], c2: closest_points[1]}}

        # Return the distances in a dictionary
        # dist = {
        #     'near_dist': {'distance': near_dist[0], 'reference_points': {c1: near_dist[1], c2: near_dist[2]}},
        #     'center_dist': {'distance': center_dist[0], 'reference_points': {c1: center_dist[1], c2: center_dist[2]}},
        #     'average_dist': {'distance': average_dist, 'reference_points': {c1: near_dist[1], c2: near_dist[2]}}
        # }

        return dist

    def get_indices_of_k_smallest(self, k, matrix, sorted=False):
        idx = np.argpartition(matrix.ravel(), k)
        ind = np.array(np.unravel_index(idx, matrix.shape))[:, range(min(k, 0), max(k, 0))]
        if sorted:
            values = matrix[tuple(ind)]
            xx = np.argsort(values)
            return ind[:, xx]
        else:
            return ind
        # if you want it in a list of indices . . .
        # return np.array(np.unravel_index(idx, arr.shape))[:, range(k)].transpose().tolist()

    def merge_clusters(self, cluster1, cluster2):
        '''
        Merge two clusters. There are three cases: 1) point-point; 2) point-cluster; 3) cluster-cluster.

        :param cluster1:
        :param cluster2:
        :return:
        '''
        print('\nExisting clusters: ', len(self.initial_clusters), f'Merging clusters: {cluster1} and {cluster2}')
        is_merge, lead_cluster, child_cluster = self.vision_generic(cluster1, cluster2)
        if is_merge:
            self.history_info['inter_intra_dist_ratio'].append(1)
            print(f"Succeed. Current clusters info: {lead_cluster} and {child_cluster} merged!")
            self.merge_2_cluster(lead_cluster, child_cluster)
            return True, lead_cluster
        print('Failed.')
        return False, None

    def update_clusters(self, new_c, old_c):
        del self.initial_clusters[old_c]
        self.DIST_MATRIX[:, old_c] = np.inf
        self.DIST_MATRIX[old_c, :] = np.inf

        # Update clusters distance
        # Get all existing clusters from initial_clusters (excluding new_c itself)
        existing_clusters = [c for c in self.initial_clusters.keys() if c != new_c]
        for c in existing_clusters:
            key = frozenset((new_c, c))
            _d = self.compute_2_clusters_dist(new_c, c)
            self.clusters_dist[key] = _d
            self.DIST_MATRIX[new_c][c], self.DIST_MATRIX[c][new_c] = _d['distance'], _d['distance']
        self.MIN_BTN_CLUSTER_DIST = max(self.DIST_MATRIX.min(), self.MIN_BTN_CLUSTER_DIST)

    def merge_2_cluster(self, cluster1_id, cluster2_id):
        cluster1, cluster2 = self.initial_clusters[cluster1_id], self.initial_clusters[cluster2_id]
        p1, p2 = \
        self.clusters_dist[frozenset((cluster1_id, cluster2_id))]['reference_points'][cluster1_id], \
        self.clusters_dist[frozenset((cluster1_id, cluster2_id))]['reference_points'][cluster2_id]
        # Update cluster data
        cluster1['traces'].append((p1, p2))
        cluster1['traces'].extend(cluster2['traces'])
        cluster1['data'].extend(cluster2['data'])
        cluster1['center'] = np.mean(self.X[cluster1['data']], axis=0)

        dist_data = np.linalg.norm(self.X[cluster1['data']] - cluster1['center'], axis=1)
        cluster1['std_dist'] = np.std(dist_data)
        cluster1['past_std'].append(cluster1['std_dist'])
        cluster1['past_dists'].extend(cluster2['past_dists'])

        self.update_clusters(cluster1_id, cluster2_id)

    def vision_generic(self, cluster1, cluster2):
        proximity, distance, lead_cluster, child_cluster = self.compute_proximity(cluster1, cluster2)

        adaptive_proximity_T1, adaptive_proximity_T2, shape_diff, adp_prox = self.compute_adaptive_threshold(self.NUM_NEAREST_CLUSTERS,
                                                               lead_cluster, child_cluster, distance, proximity)
        # adaptive_proximity_T1, adaptive_proximity_T2, shape_diff = 1, 1, 1
        print(f'--Adaptive threshold: {proximity}, {adaptive_proximity_T1}, {adaptive_proximity_T2}')

        # if proximity > adaptive_proximity_T:  # self.THRESHOLD_ROXIMITY
        # adaptive_proximity_T1, adaptive_proximity_T2 = 1.6, 1.6
        if proximity > adaptive_proximity_T1 or proximity > adaptive_proximity_T2:
            self.continuation_data['merged'].append(False)
            return False, cluster1, cluster2
        else:
            self.initial_clusters[lead_cluster]['past_dists'].append(distance)
            self.initial_clusters[lead_cluster]['merging_dists'].append(distance)
            return True, lead_cluster, child_cluster

    def compute_force_btn_clusters(self, cluster1, cluster2):
        dist = self.clusters_dist[frozenset((cluster1, cluster2))]['distance']
        m1, m2 = len(self.initial_clusters[cluster1]['data']), len(self.initial_clusters[cluster2]['data'])
        f = m1 * m2 / (dist**2)
        return f

    def compute_adaptive_threshold(self, k, cluster1, cluster2, distance, proximity):
        size1, size2 = len(self.initial_clusters[cluster1]['data']), len(self.initial_clusters[cluster2]['data'])
        # Step 1 select contextual clusters
        k = min(len(self.initial_clusters) - 1, k)
        nearest_clusters1 = self.get_indices_of_k_smallest(k, self.DIST_MATRIX[cluster1, :], sorted=True)
        nearest_clusters2 = self.get_indices_of_k_smallest(k, self.DIST_MATRIX[cluster2, :], sorted=True)

        nearest_dists1 = self.DIST_MATRIX[cluster1, nearest_clusters1][0]
        nearest_dists2 = self.DIST_MATRIX[cluster2, nearest_clusters2][0]
        nearest_dists1 = np.delete(nearest_dists1, np.where(nearest_dists1 == np.inf))
        nearest_dists2 = np.delete(nearest_dists2, np.where(nearest_dists2 == np.inf))

        mean_e, std_e = np.mean([*nearest_dists1, *nearest_dists2]), np.std([*nearest_dists1, *nearest_dists2])

        base_force = self.compute_force_btn_clusters(cluster1, cluster2)
        forces = [(1, cluster2)]
        for c in nearest_clusters1[0]:
            if c in nearest_clusters2[0]:
                _force1 = self.compute_force_btn_clusters(c, cluster1)
                _force2 = self.compute_force_btn_clusters(c, cluster2)
                forces.append((math.sqrt(_force1 * _force2) / base_force, c))

        if len(forces):
            forces1 = sorted(forces, key=lambda x: x[0], reverse=True)
            cont_clusters = []
            N_rc = 2
            Total_affect = 0
            for rc in forces1:
                # if len(self.initial_clusters[rc[1]]['data']) >= max(size1, size2):
                #     cont_clusters.append(rc[1])
                #     if len(cont_clusters) == 2:
                #         break
                dist_c_cluster1 = self.clusters_dist[frozenset((rc[1], cluster1))]['distance']
                dist_c_cluster2 = self.clusters_dist[frozenset((rc[1], cluster2))]['distance'] \
                        if rc[1] != cluster2 else dist_c_cluster1
                cont_clusters.append((rc[0], (dist_c_cluster2 + dist_c_cluster1) / 2))
                Total_affect += rc[0]
                if len(cont_clusters) == N_rc:
                    break

            vision_scale = 0
            for rc in cont_clusters:
                dist_ratio = distance / rc[1] if rc[1] != 0 else 1
                _vision_scale = 2 / (1 + math.e ** (5 * dist_ratio)) + 1
                vision_scale += _vision_scale * rc[0] / Total_affect

        else:
            vision_scale = 1
            dist_ratio, dist_c_cluster1, dist_c_cluster2 = 1, 1, 1

        if len(self.initial_clusters[cluster1]['past_dists']) > 3:
            mean1, std1 = np.mean(self.initial_clusters[cluster1]['past_dists']), \
                          np.std(self.initial_clusters[cluster1]['past_dists'])
            threshold1 = 3 / (1 + math.e ** (0.3 * mean1 / std1)) + 1.4 if std1 != 0 and mean1 != 0 else 2.7
        else:
            threshold1 = 2.7
        threshold1 = vision_scale*threshold1

        if len(self.initial_clusters[cluster2]['past_dists']) > 3:
            mean2, std2 = np.mean(self.initial_clusters[cluster2]['past_dists']), \
                          np.std(self.initial_clusters[cluster2]['past_dists'])
            # threshold2 = (1 + mean2 / std2) ** (std2 / mean2) if std2 != 0 and mean2 != 0 else 2.7
            threshold2 = 3 / (1 + math.e ** (0.3 * mean2 / std2)) + 1.4 if std2 != 0 and mean2 != 0 else 2.7
        else:
            threshold2 = 2.7
        threshold2 = vision_scale*threshold2

        N1, N2 = len(self.initial_clusters[cluster1]['past_std']), \
                 len(self.initial_clusters[cluster2]['past_std'])
        N = min(N1, N2)
        if len(self.initial_clusters[cluster2]['past_std']) > 1:
            m5 = np.mean(self.initial_clusters[cluster1]['past_std'][-N:])
            m6 = np.mean(self.initial_clusters[cluster2]['past_std'][-N:])

            Nt = N
            if N1 < N2:
                for i in range(N-1, N2):
                    if self.initial_clusters[cluster2]['past_std'][i] >= self.initial_clusters[cluster1]['past_std'][-1]:
                        Nt = i + 1
                        break

                m1 = np.mean(self.initial_clusters[cluster1]['past_std'][:N])
                m2 = np.mean(self.initial_clusters[cluster2]['past_std'][:Nt])
                std1 = np.std(self.initial_clusters[cluster1]['past_std'][:N]) * m1
                std2 = np.std(self.initial_clusters[cluster2]['past_std'][:Nt]) * m2

            else:
                for i in range(N-1, N1):
                    if self.initial_clusters[cluster1]['past_std'][i] >= self.initial_clusters[cluster2]['past_std'][-1]:
                        Nt = i + 1
                        break

                m1 = np.mean(self.initial_clusters[cluster1]['past_std'][:Nt])
                m2 = np.mean(self.initial_clusters[cluster2]['past_std'][:N])
                std1 = np.std(self.initial_clusters[cluster1]['past_std'][:Nt]) * m1
                std2 = np.std(self.initial_clusters[cluster2]['past_std'][:N]) * m2

            most_same_std = round(min(std1, std2) / max(std1, std2), 3) if std1 != 0 and std2 != 0 else 1

            # std3 = np.std(self.initial_clusters[cluster1]['past_std'][:N]) * m3
            # std4 = np.std(self.initial_clusters[cluster2]['past_std'][:N]) * m4
            std5 = np.std(self.initial_clusters[cluster1]['past_std'][-N:]) * m5
            std6 = np.std(self.initial_clusters[cluster2]['past_std'][-N:]) * m6

            l_same_std = round(min(std5, std6) / max(std5, std6), 3) if std5 != 0 and std6 != 0 else 1

        else:
            self.continuation_data['mean std'].append(0)
            self.continuation_data['std std'].append(0)
            self.continuation_data['std ratio'].append(1)
            std1, std2, std3, std4, std5, std6 = 0, 0, 0, 0, 0, 0
            same_std, l_same_std, most_same_std = 1, 1, 1

        _diff = max([l_same_std, most_same_std])
        shape_diff = _diff / (1 + _diff) + 0.5 if N > 4 else 1
        threshold1 *= shape_diff
        threshold2 *= shape_diff
        return threshold1, threshold2, shape_diff, threshold1 * size1 / (size1 + size2) + \
               threshold2 * size2 / (size1 + size2)

    def compute_proximity(self, cluster1, cluster2):
        distance = self.clusters_dist[frozenset((cluster1, cluster2))]['distance']
        n1, n2 = len(self.initial_clusters[cluster1]['past_dists']), len(self.initial_clusters[cluster2]['past_dists'])
        n = max(n1//2, n2) if n1 > n2 else max(n2//2, n1)

        mean_past_dist = np.mean([*self.initial_clusters[cluster1]['past_dists'][-n:],
                                  *self.initial_clusters[cluster2]['past_dists'][-n:]]) \
            if n1 > 2 or n2 > 2 else self.MIN_BTN_CLUSTER_DIST

        if len(self.initial_clusters[cluster1]['data']) >= len(self.initial_clusters[cluster2]['data']):
            return distance / mean_past_dist, distance, cluster1, cluster2
        else:
            return distance / mean_past_dist, distance, cluster2, cluster1

    def weighted_mean_dist(self, cluster, reference_p):
        dists = []
        weight_sum = 0
        for p1, p2 in self.initial_clusters[cluster]['traces']:
            weight = max(np.linalg.norm(p1 - reference_p), np.linalg.norm(p2 - reference_p))
            # weight = 1
            dists.append((np.linalg.norm(p1 - p2), weight))
            weight_sum += 1 / weight

        weight_dist = 0
        for d in dists:
            weight_dist += d[0] * (1 / d[1]) / weight_sum
        return weight_dist

    def DFS(self, parent, adj_list, cluster, visited):
        if parent['id'] in visited:
            return
        visited.append(parent['id'])
        for child in adj_list[parent['id']]:
            child_node = {
                'data': self.initial_clusters[cluster]['data'][child],
                'id': child,
                'children': []
            }
            parent['children'].append(child_node)
            self.DFS(child_node, adj_list, cluster, visited)

    def find_closet_cluster(self, cluster, clusters_range=None):
        if clusters_range is None:
            clusters_range = self.initial_clusters.keys()
        nearest_distance = np.inf
        for k, v in self.clusters_dist.items():
            if cluster in k and list(k - {cluster})[0] in clusters_range:
                if v['distance'] < nearest_distance:
                    nearest_distance = v['distance']
                    closest_cluster = list(k - {cluster})[0]
        return closest_cluster

    def find_near_points(self, p1, p2, cluster1, cluster2):
        middle_point = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - p2)
        local_points1 = self.get_points_wthin_radius(p1, radius)
        local_points2 = self.get_points_wthin_radius(p2, radius)
        intersections = np.array([x for x in set(tuple(x) for x in local_points1) &
                                  set(tuple(x) for x in local_points2)])
        intersections = sorted([[p, np.linalg.norm(p - middle_point)] for p in intersections], key=lambda x: x[1])
        return [p[0] for p in intersections if not self.is_point_exist(p[0], self.initial_clusters[cluster1]['data']) and
                not self.is_point_exist(p[0], self.initial_clusters[cluster2]['data'])]

    def closest_points_btn_2_clusters(self, cluster1, cluster2):
        """
        Find the closest points between two clusters using a vectorized approach.

        :param cluster1: First cluster
        :param cluster2: Second cluster
        :return: Indices of the two closest points
        """
        # Get the data points for the two clusters
        data1 = self.X[self.initial_clusters[cluster1]['data']]
        data2 = self.X[self.initial_clusters[cluster2]['data']]

        # Compute the pairwise distance matrix between points in both clusters
        distance_matrix = cdist(data1, data2)

        # Find the indices of the minimum distance in the flattened distance matrix
        min_idx = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

        # Map back to the original indices in self.X
        closest_points = [
            self.initial_clusters[cluster1]['data'][min_idx[0]],
            self.initial_clusters[cluster2]['data'][min_idx[1]]
        ]

        return closest_points

    def rank_points_btn_2_clusters(self, cluster1, cluster2):
        rank_points = []
        for p1 in self.initial_clusters[cluster1]['data']:
            for p2 in self.initial_clusters[cluster2]['data']:
                dist = np.linalg.norm(p1 - p2)
                rank_points.append([[p1, p2], dist])
        return list(sorted(rank_points, key=lambda x: x[1]))

    def get_nearest_k_points(self, k, point):
        id = int(np.where((self.X == point).all(axis=1))[0][0])
        return self.points_dist[id][:k]

    def get_points_wthin_radius(self, point, radius, cluster=None, points_set=None):
        points = []
        points_set = self.points_dist[point]

        for p in points_set:
            if p[1] > radius:
                break

            if cluster is not None:
                if np.where((self.initial_clusters[cluster]['data'] == p[0]).all(axis=1))[0].shape == (0,):
                    continue
            points.append(p[0])

        return points

    def is_point_exist(self, point, group):
        if len(group):
            return not np.where((group == point).all(axis=1))[0].shape == (0,)
        else:
            return False

    def plot(self, withline=False):
        # Y_hat = []
        # X_hat = []
        # [[[Y_hat.append(int(v['label'])), X_hat.append(x)] for x in v['data']]
        #         for k, v in self.initial_clusters.items()]
        #
        # yhat, xhat = np.array(Y_hat), np.array(X_hat)
        # assign a cluster to each example
        # yhat = model.predict(X)
        # retrieve unique clusters
        # clusters = np.unique(yhat)
        f = plt.figure(figsize=(12, 12))
        if self.DIMENSION == 2:
            ax1 = f.add_subplot(2, 2, 1)
        else:
            ax1 = f.add_subplot(2, 2, 1, projection='3d')
        ax2 = f.add_subplot(2, 2, 2)
        ax3 = f.add_subplot(2, 2, 3)
        ax4 = f.add_subplot(2, 2, 4)
        # f, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(projection='3d'))

        # create scatter plot for samples from each cluster
        # for cluster in clusters:
        #     # get row indexes for samples with this cluster
        #     row_ix = np.where(yhat == cluster)
        #     # create scatter of these samples
        #     axes[0, 0].scatter(xhat[row_ix, 0], xhat[row_ix, 1])

        for k, cluster in self.initial_clusters.items():
            if self.DIMENSION == 2:
                scatter = ax1.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1])
                ax1.text(self.X[cluster['data'][0]][0], self.X[cluster['data'][0]][1], k)
                if withline:
                    for track in cluster['traces']:
                        # axes[0, 0].plot(np.array(track)[:, 0], np.array(track)[:, 1],
                        #                 color=scatter.get_facecolor()[0], linestyle='-')
                        ax1.plot(self.X[list(track)][:, 0], self.X[list(track)][:, 1], 'k-')
            else:
                scatter = ax1.scatter(self.X[cluster['data']][:, 0], self.X[cluster['data']][:, 1],
                                      self.X[cluster['data']][:, 2])
                # ax1.text(cluster['data'][0][0], cluster['data'][0][1], cluster['data'][0][2], k)
                # if withline:
                #     for track in cluster['traces']:
                #         # axes[0, 0].plot(np.array(track)[:, 0], np.array(track)[:, 1],
                #         #                 color=scatter.get_facecolor()[0], linestyle='-')
                #         ax1.plot(np.array(track)[:, 0], np.array(track)[:, 1], np.array(track)[:, 2], 'k-')

        # show the plot
        # plt.show()

        for k, v in self.initial_clusters.items():
            if len(v['merging_dists']) <= 1:
                continue
            ratio = [v['merging_dists'][i] / (sum(v['merging_dists'][: i]) / i) for i in range(1, len(v['merging_dists']))]
            ax2.plot(range(len(v['merging_dists'])), v['merging_dists'], '--.')
            ax2.plot(range(1, len(v['merging_dists'])), ratio, ':')

            ratio2 = [(v['past_densities'][i-1]) / v['past_densities'][i] for i in range(1, len(v['past_densities']))]
            ax3.plot(range(len(v['past_densities'])), v['past_densities'], '--.')
            ax3.plot(range(1, len(v['past_densities'])), ratio2, ':')

            # ratio3 = [v['past_std'][i] / (sum(v['past_std'][: i]) / i) if (sum(v['past_std'][: i]) / i) > 0.01 else 0
            #           for i in range(2, len(v['past_std']))]
            # ratio3 = [v['past_std'][i] / v['past_std'][i-1] if v['past_std'][i-1] > 0.01 else 0
            #           for i in range(1, len(v['past_std']))]
            ax4.plot(range(len(v['past_std'])), v['past_std'], '--.')
            # axes[1, 1].plot(range(1, len(v['past_std'])), ratio3, ':')

        plt.savefig(f'{self.log_path}/{len(self.initial_clusters)}.png')
        plt.close('all')
