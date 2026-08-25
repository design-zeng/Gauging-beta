import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from clt_tools import load_data

matplotlib.use("tKAgg")

ALPHA = 0.1
p = 0.02

class lower_density_selection:
    def __init__(self, points):
        self.points = points

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def find_longest_distance(self, points):
        max_distance = 0
        for point in points:
            for other_point in points:
                distance = self.calculate_distance(point, other_point)
                if distance > max_distance:
                    max_distance = distance
        return max_distance

    def find_density(self, points, radius):
        densities = []
        total_points = len(points)
        total_neighbors = 0

        for point in points:
            neighbors = 0
            for other_point in points:
                if tuple(point) == tuple(other_point):
                    continue
                distance = self.calculate_distance(point, other_point)
                if distance <= radius:
                    neighbors += 1
            densities.append(neighbors)
            total_neighbors += neighbors

        average_neighbors = total_neighbors / total_points

        return densities, average_neighbors



    def find_minimum_radius(self, points):
        low = 0.0
        high = self.find_longest_distance(points)

        required_neighbors = p * len(points)

        while high - low > 1e-6:
            mid = (low + high) / 2.0
            densities, avg_neighbors = self.find_density(points, mid)

            if avg_neighbors < required_neighbors:
                low = mid
            else:
                high = mid

        return densities, low

    def run(self):

        densities, _ = self.find_minimum_radius(self.points)

        # Calculate the first quartile (Q1) and third quartile (Q3) of the densities
        Q1 = np.percentile(densities, 25)
        Q3 = np.percentile(densities, 75)

        # Find points whose density is lower than Q1 - 0.1*(Q3 - Q1)
        low_density_points = [i for i, point in enumerate(self.points) if densities[i] < Q1 - ALPHA * (Q3 - Q1)]
        return low_density_points

    def plot_lower_density(self):
        low_density_points = self.run()

        f = plt.figure()

        # Create a scatter plot of all points
        x_points = [point[0] for point in self.points]
        y_points = [point[1] for point in self.points]
        plt.scatter(x_points, y_points, s=12)

        # Mark points with lower density in red
        x_low_density = [self.points[idx][0] for idx in low_density_points]
        y_low_density = [self.points[idx][1] for idx in low_density_points]
        plt.scatter(x_low_density, y_low_density, color='red', s=12)

        ax = plt.gca()  # Get current axis
        ax.set_xticks([])  # Remove x-axis tick labels
        ax.set_yticks([])  # Remove y-axis tick labels
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        return f


if __name__ == '__main__':
    points = load_data('data/aggregation.txt', 'agg')[0]
    lower_density_selector = lower_density_selection(points)
    fig = lower_density_selector.plot_lower_density()
    fig.savefig('agg.png', bbox_inches='tight')