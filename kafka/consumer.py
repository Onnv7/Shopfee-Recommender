import json
from kafka import KafkaConsumer
topic_name = "nhac_vang"
c = KafkaConsumer(
    topic_name,
    bootstrap_servers = ['localhost:9094'],
    auto_offset_reset = 'latest',
    enable_auto_commit = True
)


for message in c:
    print("Da nhan ", message.value)