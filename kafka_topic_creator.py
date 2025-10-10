from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

KAFKA_BROKER = "localhost:9092"

# 토픽 이름
TOPIC_NAMES = ["log-topic", "chat-topic"], 

# KafkaAdminClient를 사용해 토픽 생성
admin_client = KafkaAdminClient(
    bootstrap_servers=KAFKA_BROKER,
    client_id='topic-creator'
)

for name in TOPIC_NAMES:
    topic = NewTopic(
        name=name,
        num_partitions=1,
        replication_factor=1
    )

    try:
        admin_client.create_topics(new_topics=[topic], validate_only=False)
        print(f"토픽 '{name}' 생성")
    except TopicAlreadyExistsError:
        print(f"토픽 '{name}' 이미 존재")
    finally:
        admin_client.close()
