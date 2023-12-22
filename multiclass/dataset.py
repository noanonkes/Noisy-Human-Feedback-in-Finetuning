from torch.utils.data import Dataset
import numpy as np

 
class RLHFRatingDataset(Dataset):
    def __init__(self, embeddings, ratings, k=1):
        """
        k: int: number of samples to compare in one data point
        """
        self.embeddings = embeddings
        self.ratings = ratings

        self.pairs = self.create_pairs(ratings, k)
        
    def create_pairs(self, ratings, k):
        max_rating = np.max(ratings)
        idx = np.arange(len(ratings))
        idx_used = set()
        pairs = []

        all_idx = [idx[ratings == i] for i in range(max_rating + 1)]

        for rating_score, idx_arr in enumerate(all_idx):
            # if k = 2, stop when rating_score = 4, other wise
            # we can't make (3, 4)
            if rating_score == max_rating - (k - 1):
                break

            for idx_i in idx_arr:
                # if we have already used this one, skip this one
                if idx_i in idx_used:
                    continue

                # other wise, we will use it and add to used
                idx_used.add(idx_i)

                # want no duplicate labels
                labels_used = set([rating_score]) 
                
                # this index will be first of pair
                rating_list = [idx_i]
                
                # for how many comparisons we want in our pair
                # add one of one higher each time
                found = 1
                for higher_rating_score in range(1, k + 1):
                    for _ in range(200):
                        # randomly sample a index of a higher rating
                        idx_j = np.random.choice(np.concatenate(all_idx[rating_score + higher_rating_score:]), 1)[0]
                        r_j = ratings[idx_j]

                        # if we haven't seen it before, we can use it
                        if idx_j not in idx_used and r_j not in labels_used:
                            found += 1
                            break
                    
                    # also add this one to used
                    idx_used.add(idx_j)
                    labels_used.add(r_j)
                    rating_list.append(idx_j)

                # add this list of indices to all pairs (for lack of a better word) in DESCENDING ORDER!
                if found == k + 1:
                    pairs.append(list(sorted(rating_list, reverse=True)))
        
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        return self.embeddings[pair], self.ratings[pair] 
    

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]