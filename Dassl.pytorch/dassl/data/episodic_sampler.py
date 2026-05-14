import random
import copy
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class EpisodicSampler(Sampler):
    """Episodic sampler for N-way K-shot meta-learning.
    
    Constructs episodes where each episode contains:
    - Support set: N classes × K_support samples
    - Query set: N classes × K_query samples
    
    Args:
        data_source (list): list of Datums.
        n_way (int): number of classes per episode.
        k_support (int): number of support samples per class.
        k_query (int): number of query samples per class.
        n_episodes (int): number of episodes to generate.
    """
    
    def __init__(self, data_source, n_way, k_support, k_query, n_episodes):
        self.data_source = data_source
        self.n_way = n_way
        self.k_support = k_support
        self.k_query = k_query
        self.n_episodes = n_episodes
        
        # Build index dictionary: label -> list of indices
        self.index_dic = defaultdict(list)
        for idx, item in enumerate(data_source):
            self.index_dic[item.label].append(idx)
        
        self.labels = list(self.index_dic.keys())
        
        # Validate n_way
        if self.n_way > len(self.labels):
            raise ValueError(
                f"n_way={self.n_way} exceeds number of available labels={len(self.labels)}"
            )
        
        # Validate that each label has enough samples
        min_samples_needed = self.k_support + self.k_query
        for label in self.labels:
            num_samples = len(self.index_dic[label])
            if num_samples < min_samples_needed:
                raise ValueError(
                    f"Label {label} has only {num_samples} samples, "
                    f"but need at least k_support + k_query = {min_samples_needed}"
                )
        
        self.length = n_episodes
    
    def __iter__(self):
        """Generate episodes."""
        for _ in range(self.n_episodes):
            # Sample n_way classes
            selected_labels = random.sample(self.labels, self.n_way)
            
            support_indices = []
            query_indices = []
            
            # For each selected class, sample support and query indices
            for label in selected_labels:
                # Get all indices for this label
                label_indices = self.index_dic[label]
                
                # Sample k_support + k_query indices without replacement
                sampled_indices = random.sample(
                    label_indices, 
                    self.k_support + self.k_query
                )
                
                # Split into support and query
                support_indices.extend(sampled_indices[:self.k_support])
                query_indices.extend(sampled_indices[self.k_support:])
            
            yield support_indices, query_indices
    
    def __len__(self):
        return self.length
