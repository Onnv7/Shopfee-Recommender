import json

from kafka import KafkaProducer

topic_name = "nhac_vang"

p = KafkaProducer(
    bootstrap_servers = ['localhost:9094']
)

json_mess = json.dumps({"user_id":"U0001120", "gender": "FEMALE"})

p.send( topic_name, json_mess.encode("utf-8"))
p.flush()

print("Da gui")