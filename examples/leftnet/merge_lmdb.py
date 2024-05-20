import lmdb

env = lmdb.open("data/jarvis/")
txn = env.begin(write=True)

database1 = env.open_db("random_train")
database2 = txn.cursor("data_train")
database3 = txn.cursor("data_valid")

env.open_db(key="data_full", txn=txn)
newDatabase = txn.cursor("data_full")

for (key, value) in database1:
    newDatabase.put(key, value)

for (key, value) in database2:
    newDatabase.put(key, value)

for (key, value) in database3:
    newDatabase.put(key, value)