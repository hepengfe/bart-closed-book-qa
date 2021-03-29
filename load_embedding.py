
# embed dim = 768, cluster 100 passages 
# pick these cluter centers and rank them and get top 3 clusters out of K clusters

# how to detemrine K ? 
# But intuitively, suppose there are 7 passages per cluster (ideally due to the input length limit). 
# then it would be 100/7= 14 clusters
# 6-10 clusters should be reasonable, and we just take top 3 clusters 

# take 10 clusters, output 10~20 answers, and then set the threshold for the 10 ~ 20 answers
# cluster center threshold to filter out some of those clusters 
# in this way, we basically considered all passages and set thresholding to filter out them



# dataset requirement:  one question generate c QP concatenation -> and all QP concatenation for checking answer coverage

if __name__ == "__main__":
    import pickle
    embedding_path = "data/wiki_embeddings/"
    with open(embedding_path + 'wikipedia_passages_0.pkl', 'rb') as f:
        data = pickle.load(f)
    import pdb; pdb.set_trace()
