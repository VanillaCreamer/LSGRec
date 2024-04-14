# 从txt文件中读取用户-物品-评分的交互数据
def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            user, item, rating = line.strip().split(',')
            data.append((user, item, float(rating)))
    return data

# 将用户-物品对分为喜欢和不喜欢两个集合
def split_likes(data):
    likes = {}
    dislikes = {}
    for user, item, rating in data:
        if rating > 2.5:
            if user not in likes:
                likes[user] = set()
            likes[user].add(item)
        else:
            if user not in dislikes:
                dislikes[user] = set()
            dislikes[user].add(item)
    return likes, dislikes

# 对于用户u，找到所有与他有共同喜欢物品的用户
def find_similar_users(user, likes):
    similar_users = []
    for u, items in likes.items():
        if u != user and len(items.intersection(likes[user])) > 0:
            similar_users.append(u)
    return similar_users

# 计算用户u不喜欢的物品在这个集合中命中了多少
def calculate_hit_count(user, dislikes, similar_users):
    hit_count = 0
    similar_dislike = set()
    # for u in similar_users:
    #     if u in dislikes and len(dislikes[u].intersection(likes[user])) > 0:
    #         hit_count += 1
    for u in similar_users:
        if u in dislikes:
            similar_dislike = similar_dislike | dislikes[u]
    for neg in dislikes[user]:
        if neg in similar_dislike:
            hit_count += 1

    return hit_count / len(dislikes[user])


import torch
import torch.nn.functional as F


def calculate_similarity(user, dislikes, similar_users, embeds):
    hit_count = 0
    similar_dislike = set()
    for u in similar_users:
        if u in dislikes:
            similar_dislike = similar_dislike | dislikes[u]

    print(f'There are {len(similar_dislike)} items in similar dislike set.')
    # item = embeds[list(map(int, list(similar_dislike)))]
    # PyTorch 处理余弦相似度
    # batch_of_vectors = embeds[list(map(int, list(dislikes[user])))]
    batch_of_vectors = embeds[list(map(int, list(similar_dislike)))]
    similarity_matrix = F.cosine_similarity(batch_of_vectors.unsqueeze(1), batch_of_vectors.unsqueeze(0), dim=2)
    similarity_matrix.fill_diagonal_(0)
    sorted_matrix, _ = torch.sort(similarity_matrix.reshape(-1), descending=True)
    mean_sim = torch.mean(sorted_matrix[:50])

    # distances = torch.cdist(item, item, p=2)
    # mean_dis = torch.mean(distances)
    return mean_sim


file_path = './data/amazon-book/train_yelp1.dat'
data = read_data(file_path)
likes, dislikes = split_likes(data)
# item_emb = torch.load('./checkpoints/yelp_our_item.pth')

# user_u = '1'
# similar_users = find_similar_users(user_u, likes)
# print(f'There are {len(similar_users)} users like at least one same item which {user_u} likes.')
# hit_ratio = calculate_hit_count(user_u, dislikes, similar_users)
# print(f"In user{user_u}'s dislike items, there are {100*hit_ratio}% items that his/her similar users dislike, i.e., positive edges pass {100*hit_ratio}% negative preference.")
# mean_similarity = calculate_similarity(user_u, dislikes, similar_users, item_emb)
# print(f'The average of the similarity between items in the similar dislike items is {mean_similarity}')


# 随机采样1000个
import random

# 随机采样1000个用户u
# random_users = random.sample(list(dislikes.keys()), 200)
random_users = random.sample(set(dislikes) & set(likes), 1000)

total_hit_ratio = 0
for user_u in random_users:
    similar_users = find_similar_users(user_u, likes)
    hit_ratio = calculate_hit_count(user_u, dislikes, similar_users)
    total_hit_ratio += hit_ratio

average_hit_count = total_hit_ratio / len(random_users)
print(f"Average hit count for 1000 random users: {average_hit_count}")




