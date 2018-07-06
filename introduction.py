from __future__ import division

#***************************************************************************************************
# **********************Chapter 1. Introduction****************************************************
#***************************************************************************************************

users = [
{ "id": 0, "name": "Hero" },
{ "id": 1, "name": "Dunn" },
{ "id": 2, "name": "Sue" },
{ "id": 3, "name": "Chi" },
{ "id": 4, "name": "Thor" },
{ "id": 5, "name": "Clive" },
{ "id": 6, "name": "Hicks" },
{ "id": 7, "name": "Devin" },
{ "id": 8, "name": "Kate" },
{ "id": 9, "name": "Klein" }
]

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

#1. For example, we might want to add a list of friends to each user. 
# First we set each user’s friends property to an empty list:
for user in users:
    user["friends"] = []

for i, j in friendships:
    # this works because users[i] is the user whose id is i
    users[i]["friends"].append(users[j]) 	# add i as a friend of j
    users[j]["friends"].append(users[i]) 	# add j as a friend of i

#2.
# Once each user dict contains a list of friends, we can easily ask questions of our graph, like "what’s the average 
# number of connections?" First we find the total number of connections, by summing up the lengths of all the friends lists:
def number_of_friends(user):
    """how many friends does _user_ have?"""
#    print len(user["friends"])
    return len(user["friends"])					         # length of friend_ids list
total_connections = sum(number_of_friends(user) for user in users) 	 # 24
# print 'total_connections are %r ' %total_connections

# And then we just divide by the number of users:
num_users = len(users) 					# length of the users list
avg_connections = total_connections / num_users	# 2.4
# print 'avg_connections are %r' %avg_connections


#3.
# It’s also easy to find the most connected people - they’re the people who have the largest number of friends.
# Since there aren’t very many users, we can sort them from "most friends" to "least friends":
# create a list (user_id, number_of_friends)
num_friends_by_id=[(user['id'],number_of_friends(user)) for user in users]
# print sorted(num_friends_by_id,          # get it sorted
#             key=lambda x:x[1],          # by num_friends
#             reverse=True)               # largest to smallest         


#4.
#VP asks you to design a "Data Scientists You May Know" suggester. Your first instinct is to suggest that a user might know 
# the friends of friends. These are easy to compute: for each of a user’s friends, iterate over that person’s friends, 
# and collect all the results:
def friends_of_friend_ids_bad(user):
    # "foaf" is short for "friend of a friend"
    return [foaf['id']
            for friend in user['friends']    # for each of user's friends
            for foaf in friend['friends']]   # get each of _their_ friends
# print friends_of_friend_ids_bad(users[0]) 


# ???? skipped the remaining chapter right now. However should be completed later.
