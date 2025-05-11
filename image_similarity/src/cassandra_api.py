import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from ssl import SSLContext, PROTOCOL_TLSv1_2 , CERT_REQUIRED
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import boto3
import json
class Cassandra:
    # def __init__(self, address, port):
    #     profile = ExecutionProfile(
    #     load_balancing_policy=WhiteListRoundRobinPolicy([str(address)]),
    #     retry_policy=DowngradingConsistencyRetryPolicy(),
    #     consistency_level=ConsistencyLevel.LOCAL_QUORUM,
    #     serial_consistency_level=ConsistencyLevel.LOCAL_SERIAL,
    #     request_timeout=15,
    #     row_factory=tuple_factory
    #     )

    #     self.cluster = Cluster([str(address)], port=port, execution_profiles={EXEC_PROFILE_DEFAULT:profile})

    def __init__(self):

        file_path = r'C:\keys\keyspace\application.json'

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        USERNAME = data["username"]
        PASSWORD = data["password"]

        ssl_context = SSLContext(PROTOCOL_TLSv1_2)
        ssl_context.load_verify_locations('c:/keys/keyspace/sf-class2-root.crt')
        ssl_context.verify_mode = CERT_REQUIRED
        auth_provider = PlainTextAuthProvider(username=USERNAME, password=PASSWORD)
        self.cluster = Cluster(['cassandra.ap-northeast-2.amazonaws.com'], ssl_context=ssl_context, auth_provider=auth_provider, port=9142)

    
    def load_all_user(self):
        """
        Load all user_id with user profile image path
        
        Return: List[user_id, image_path]
        """
        session = self.cluster.connect()
        rows = session.execute("SELECT * FROM test.testml")
        result_dicts = [{"user_name": row.user_name, "image_path":row.image_path} for row in rows]
        res = [(user_row.user_name, user_row.image_path) for user_row in rows]
        


        return result_dicts