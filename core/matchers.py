import math
from abc import ABC, abstractmethod
from typing import List, Tuple
from numpy.random import randint, rand
from core.fingerprint import Fingerprint, Minutia
import numpy as np

class BaseMatcher(ABC):
    @abstractmethod
    def match(self, probe: Fingerprint, gallery: Fingerprint) -> float:
        """Returns a matching similarity score."""
        pass

class HoughMatcher(BaseMatcher):
    # Thresholds adapted from the original assignment logic
    DISTANCE_THRESHOLD = 10
    ROTATION_THRESHOLD = 20
    
    def match(self, probe: Fingerprint, gallery: Fingerprint) -> float:
        """Matches using Generalized Hough Transform alignment[cite: 2, 3]."""
        if not probe.minutiae or not gallery.minutiae:
            return 0.0

        transform_params = self._hough_transform(probe.minutiae, gallery.minutiae)
        if not transform_params:
            return 0.0
            
        matched_pairs = self._minutiae_pairing(probe.minutiae, gallery.minutiae, transform_params)
        
        # Feature 1 represents the raw count of matched minutiae
        feature_1 = len(matched_pairs)
        feature_2 = 0  # Placeholder for secondary feature (e.g., area bounding)
        
        score = 0.9 * feature_1 + 0.1 * feature_2
        return score

    def _hough_transform(self, Q_set: List[Minutia], T_set: List[Minutia]):
        A = {}
        max_val, max_val_key = -float('inf'), None
        
        for q in Q_set:
            for t in T_set:
                del_theta = t.theta - q.theta
                rad_del_theta = math.radians(del_theta)
                
                del_x = t.x - q.x * math.cos(rad_del_theta) - q.y * math.sin(rad_del_theta)
                del_y = t.y + q.x * math.sin(rad_del_theta) - q.y * math.cos(rad_del_theta)
                
                k = (int(del_theta), int(del_x), int(del_y))
                A[k] = A.get(k, 0) + 1
                
                if A[k] > max_val:
                    max_val = A[k]
                    max_val_key = k
        return max_val_key

    def _minutiae_pairing(self, Q_set: List[Minutia], T_set: List[Minutia], transform_params):
        del_theta, del_x, del_y = transform_params
        f_T = [False] * len(T_set)
        f_Q = [False] * len(Q_set)
        ret = []
        
        # 1. Convert global rotation to radians
        rad_theta = math.radians(del_theta)
        
        for i, q in enumerate(Q_set):
            # 2. Apply global Hough transformation to align query minutia
            aligned_theta = q.theta + del_theta
            
            # Using the inverse of the hough transform geometry to map Q to T space
            aligned_x = del_x + q.x * math.cos(rad_theta) + q.y * math.sin(rad_theta)
            aligned_y = del_y - q.x * math.sin(rad_theta) + q.y * math.cos(rad_theta)
            
            for j, t in enumerate(T_set):
                if not f_T[j] and not f_Q[i]:
                    # 3. Calculate distance strictly between ALIGNED query and template
                    dist = math.sqrt((t.x - aligned_x)**2 + (t.y - aligned_y)**2)
                    theta_diff = abs(t.theta - aligned_theta)
                    
                    if dist < self.DISTANCE_THRESHOLD and theta_diff < self.ROTATION_THRESHOLD:
                        f_T[j] = True
                        f_Q[i] = True
                        ret.append((j, i))
                        break # Move to the next query minutia once a match is found
        return ret


class GeneticMatcher(BaseMatcher):
    def __init__(self, n_iter=100, n_pop=100, r_cross=0.9, thold=12, early_stop_thresh=50.0):
        """Optimized Genetic Algorithm with restored population sizing and stricter distance thresholds."""
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.r_cross = r_cross
        self.r_mut = 1.0 / (27.0 * 2.0)
        self.thold = thold
        self.early_stop_thresh = early_stop_thresh

    def _fitness_function(self, s, theta, tx, ty, mx, my, qx, qy):
        theta_rad = math.radians(theta)
        sin_theta = math.sin(theta_rad)
        cos_theta = math.cos(theta_rad)

        # 1. Vectorized spatial transformation of the entire gallery set at once
        transformed_x = s * (mx * cos_theta - my * sin_theta) + tx
        transformed_y = s * (mx * sin_theta + my * cos_theta) + ty

        # 2. Matrix broadcasting to compute all pairwise distances instantly
        # This creates a 2D grid of distances between every gallery and query point
        dx = transformed_x[:, np.newaxis] - qx
        dy = transformed_y[:, np.newaxis] - qy
        
        # Calculate squared distances (avoids the expensive math.sqrt operation)
        distances_sq = dx**2 + dy**2
        
        # 3. Create a boolean matrix of matches that fall within the squared threshold
        valid_matches = distances_sq < (self.thold ** 2)
        
        # 4. Collapse the matrix: Check if a query point matched AT LEAST ONE gallery point
        matched_queries = np.any(valid_matches, axis=0)
        
        return np.sum(matched_queries)

    def match(self, probe: Fingerprint, gallery: Fingerprint) -> float:
        """Matches using Generalized Hough Transform alignment."""
        if not probe.minutiae or not gallery.minutiae:
            return 0.0

        transform_params = self._hough_transform(probe.minutiae, gallery.minutiae)
        if not transform_params:
            return 0.0
            
        matched_pairs = self._minutiae_pairing(probe.minutiae, gallery.minutiae, transform_params)
        
        match_count = len(matched_pairs)
        
        # NORMALIZE THE SCORE:
        # Divide the matched count by the total possible matches (the smaller of the two minutiae sets)
        # This prevents dense/noisy fingerprints from winning via random probability
        min_minutiae = min(len(probe.minutiae), len(gallery.minutiae))
        
        if min_minutiae == 0:
            return 0.0
            
        # Convert to a 0-100 percentage score
        score = (match_count / min_minutiae) * 100.0
        
        return score

    def _convert_list_to_integer(self, lst):
        return sum(val * (2**idx) for idx, val in enumerate(lst))

    def _get_value_from_chromosome(self, single_chromo):
        s_list = single_chromo[:5]
        theta_list = single_chromo[5:11]
        tx_list = single_chromo[11:19]
        ty_list = single_chromo[19:27]
        
        s = self._convert_list_to_integer(s_list) * 0.01 + 0.9
        theta = self._convert_list_to_integer(theta_list) - 30
        tx = self._convert_list_to_integer(tx_list) - 128
        ty = self._convert_list_to_integer(ty_list) - 128
        return s, theta, tx, ty

    def _selection(self, pop, scores, k=3):
        selection_ix = randint(len(pop))
        for ix in randint(0, len(pop), k-1):
            if scores[ix] > scores[selection_ix]: 
                selection_ix = ix
        return pop[selection_ix]

    def _crossover(self, p1, p2):
        c1, c2 = p1.copy(), p2.copy()
        if rand() < self.r_cross:
            pt = randint(1, len(p1)-2)
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def _mutation(self, bitstring):
        for i in range(len(bitstring)):
            if rand() < self.r_mut:
                bitstring[i] = 1 - bitstring[i]

    def _genetic_algorithm(self, mx, my, qx, qy):
        pop = [randint(0, 2, 27).tolist() for _ in range(self.n_pop)]
        s, theta, tx, ty = self._get_value_from_chromosome(pop[0])
        best, best_eval = pop[0], self._fitness_function(s, theta, tx, ty, mx, my, qx, qy)
        
        for gen in range(self.n_iter):
            scores = [self._fitness_function(*self._get_value_from_chromosome(p), mx, my, qx, qy) for p in pop]
            
            for i in range(self.n_pop):
                if scores[i] > best_eval:
                    best, best_eval = pop[i], scores[i]
                    
            # EARLY STOPPING: Break immediately if we find a definitive match to save compute time
            if best_eval >= self.early_stop_thresh:
                break
                    
            selected = [self._selection(pop, scores) for _ in range(self.n_pop)]
            children = []
            
            for i in range(0, self.n_pop, 2):
                p1, p2 = selected[i], selected[i+1]
                for c in self._crossover(p1, p2):
                    self._mutation(c)
                    children.append(c)
            pop = children
            
        return best, best_eval