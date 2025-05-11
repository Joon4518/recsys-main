import cassandra
from typing import Union
from fastapi import FastAPI

app = FastAPI()

from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.policies import WhiteListRoundRobinPolicy, DowngradingConsistencyRetryPolicy
from cassandra.query import tuple_factory
from cassandra.query import ConsistencyLevel

profile = ExecutionProfile(
    load_balancing_policy=WhiteListRoundRobinPolicy(["61.80.148.154"]),
    retry_policy=DowngradingConsistencyRetryPolicy(),
    consistency_level=ConsistencyLevel.LOCAL_QUORUM,
    serial_consistency_level=ConsistencyLevel.LOCAL_SERIAL,
    request_timeout=15,
    row_factory=tuple_factory
)

TOP_N = 10
cluster = Cluster(["61.80.148.154"], port=30003, execution_profiles={EXEC_PROFILE_DEFAULT : profile})

@app.get("/recommender/{user_id}")
async def get_recommends(user_id: str):
    res = []

    session = cluster.connect()
    
    # rating_test table's scheme : user_id, rating, target_id
    rating_query = session.prepare("SELECT * FROM user_profile_test.rating_test WHERE user_id=?")
    
    # Results are sorted by Cluster key(rating)
    user_favored_list = [target_id for _, _, target_id in session.execute(rating_query, [int(user_id)])]
    
    # similarity_test table's scheme : user_id, similarity, target_id
    similarity_query = "SELECT * FROM user_profile_test.similarity_test WHERE user_id=%s"
    similarity_query_futures = [session.execute_async(similarity_query, [target_id]) for target_id in user_favored_list[:min(len(user_favored_list), 5)]]

    # wait for them to complete and use the results
    for future in similarity_query_futures:
        rows = future.result()
        for _, _, similar_user_id in rows:
            if len(res) == TOP_N:
                break
            # TODO: Add filter function
            if filter(similar_user_id):
                continue
            res.append(similar_user_id)

    return res

def filter(similar_user_id):
    if similar_user_id in ["History"]:
        return True