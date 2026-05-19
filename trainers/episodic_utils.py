"""Shared episodic utilities for CoOp+ATP meta-learning variants."""

import torch
import numpy as np


class EpisodicTaskSampler:
    """Sample N-way K-shot episodes from a dataset.
    
    Args:
        data_source: list of dataset items with .label attribute
        n_way: number of classes per episode
        k_support: number of support samples per class
        k_query: number of query samples per class
        n_episodes: total number of episodes to generate
    """
    def __init__(self, data_source, n_way=20, k_support=3, k_query=10, n_episodes=100):
        self.data_source = data_source
        self.n_way = n_way
        self.k_support = k_support
        self.k_query = k_query
        self.n_episodes = n_episodes
        
        # Group indices by class
        self.class_indices = {}
        for idx, item in enumerate(data_source):
            label = item.label
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        self.available_classes = sorted(self.class_indices.keys())
    
    def sample_episode(self):
        """Sample one N-way K-shot episode.
        
        Returns:
            support_idxs: list of indices for support set
            query_idxs: list of indices for query set
        """
        import random
        # Sample N classes
        classes = random.sample(self.available_classes, self.n_way)
        
        support_idxs = []
        query_idxs = []
        
        for cls in classes:
            cls_indices = self.class_indices[cls]
            # Sample K_support + K_query samples
            n_needed = self.k_support + self.k_query
            if len(cls_indices) < n_needed:
                # With replacement if not enough samples
                sampled = random.choices(cls_indices, k=n_needed)
            else:
                sampled = random.sample(cls_indices, n_needed)
            
            support_idxs.extend(sampled[:self.k_support])
            query_idxs.extend(sampled[self.k_support:])
        
        return support_idxs, query_idxs
    
    def __iter__(self):
        for _ in range(self.n_episodes):
            yield self.sample_episode
    
    def __len__(self):
        return self.n_episodes


def attribute_similarity_matrix(classnames, attr_embeddings):
    """Compute cosine similarity matrix between class attribute embeddings.
    
    Args:
        classnames: list of class names
        attr_embeddings: tensor of shape (n_classes, attr_dim) — pre-computed attribute embeddings
    
    Returns:
        similarity_matrix: tensor of shape (n_classes, n_classes) with cosine similarities
    """
    # Normalize embeddings
    normed = attr_embeddings / attr_embeddings.norm(dim=-1, keepdim=True)
    # Cosine similarity
    sim_matrix = normed @ normed.t()
    return sim_matrix


def sample_hard_episode(classnames, attr_embeddings, n_way, k_support, k_query,
                        class_indices, tau=0.5, hard_ratio=0.5):
    """Sample a hard-transfer episode based on attribute similarity.
    
    Hard episode: classes are different but have similar attributes (e.g., wolf/husky/fox share fur/tail/canine).
    This enhances compositional reasoning.
    
    Args:
        classnames: list of class names
        attr_embeddings: tensor of shape (n_classes, attr_dim)
        n_way: number of classes per episode
        k_support: support samples per class
        k_query: query samples per class
        class_indices: dict mapping class label -> list of data indices
        tau: similarity threshold (classes with sim > tau are considered similar)
        hard_ratio: probability of sampling a hard episode vs random
    
    Returns:
        support_idxs, query_idxs
    """
    import random
    
    sim_matrix = attribute_similarity_matrix(classnames, attr_embeddings)
    n_classes = len(classnames)
    available_classes = list(range(n_classes))
    
    # Decide hard vs random
    if random.random() < hard_ratio:
        # Hard episode: pick anchor, find similar classes
        anchor = random.choice(available_classes)
        # Find classes with similarity > tau (excluding self)
        similar = [j for j in available_classes if j != anchor and sim_matrix[anchor, j] > tau]
        
        if len(similar) >= n_way - 1:
            # Enough similar classes
            episode_classes = [anchor] + random.sample(similar, n_way - 1)
        else:
            # Not enough similar classes, fall back to random
            episode_classes = random.sample(available_classes, n_way)
    else:
        # Random episode
        episode_classes = random.sample(available_classes, n_way)
    
    support_idxs = []
    query_idxs = []
    
    for cls in episode_classes:
        cls_indices = class_indices[cls]
        n_needed = k_support + k_query
        if len(cls_indices) < n_needed:
            sampled = random.choices(cls_indices, k=n_needed)
        else:
            sampled = random.sample(cls_indices, n_needed)
        
        support_idxs.extend(sampled[:k_support])
        query_idxs.extend(sampled[k_support:])
    
    return support_idxs, query_idxs
