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
            # if rating_score == 3:
            #     break
            # if k = 2, stop when rating_score = 4, other wise
            # we can't make (3, 4)
            if rating_score == max_rating - (k - 1):
                break

            for i in idx_arr:
                # if we have already used this one, skip this one
                if i in idx_used:
                    continue

                # other wise, we will use it and add to used
                idx_used.add(i)
                
                # this index will be first of pair
                rating_list = [i]
                
                # for how many comparisons we want in our pair
                # add one of one higher each time
                for higher_rating_score in range(1, k + 1):
                    # print("rating + higher:", rating_score, higher_rating_score)
                    while True:
                        # inter = rating_score + 2 * higher_rating_score
                        # print("inter", inter)

                        # randomly sample a index of a higher rating
                        # j = np.random.choice(all_idx[inter], 1)[0]
                        j = np.random.choice(all_idx[rating_score + higher_rating_score], 1)[0]

                        # if we haven't seen it before, we can use it
                        if j not in idx_used:
                            break
                    
                   # also add this one to used
                    idx_used.add(j)
                    rating_list.append(j)

                # add this list of indices to all pairs (for lack of a better word)
                pairs.append(list(reversed(rating_list)))
        
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        return self.embeddings[pair], self.ratings[pair] 
    

class RLHFCarDataset(Dataset):
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.pairs = self.create_pairs(y, k)
        
    def create_pairs(self, y, k):
        max_l = np.max(y) # 3
        idx = np.arange(len(y))
        idx_used = set()
        pairs = []

        all_idx = [idx[y == i] for i in range(max_l, -1, -1)]

        for l, idx_arr in enumerate(all_idx):
            # so it's 3, 2, 1, 0
            r_l = max_l - l

            if r_l == 0:
                break

            for idx_i in idx_arr:

                # if we have already used this one, skip this one
                if idx_i in idx_used:
                    continue

                # other wise, we will use it and add to used
                idx_used.add(idx_i)
                
                # this index will be first of pair
                rating_list = [idx_i]
                
                # for how many comparisons we want in our pair
                # add one of one higher each time
                for n in range(1, k + 1):
                    # stop trying after some time
                    count = 0
                    # n = 1, 2, 3... (only 1 for now)

                    while True:
                        idx_j = np.random.choice(np.concatenate(all_idx[l + n:]), 1)[0]
                        count += 1

                        # if we haven't seen it before, we can use it
                        if idx_j not in idx_used or count == 200:
                            break
                    
                    # also add this one to used
                    idx_used.add(idx_j)
                    rating_list.append(idx_j)

                # add this list of indices to all pairs, descending order
                pairs.append(list(rating_list))
        
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        return self.X[pair], self.y[pair] 
   

class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]