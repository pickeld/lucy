from time import sleep
import requests
from samples import json_msg


conversation = [
    "?? Hi, my name is David",
    "?? What is my name?",
    "?? i was born on 9th of February 1986",
    "?? What is my birth date?",
    "?? What is my age?",
    "?? My kids are called Mia and Ben",
    "?? What are my kids names?",
]


def send_msg(msg):
    response = requests.post(
        "http://localhost:5002/webhook",
        json={
            "payload": msg
        }
    )
    return response

# for sentence in conversation:
#     print(f"Sending: {sentence}")
#     response = requests.post(
#         "http://localhost:5002/webhook",
#         json={
#             "payload":
#                 about:blank{
#                     "fromMe": True,
#                     "body":sentence,
#                     "to": "120363401685799472@g.us",
#                 }

#         }
#     )
#     print(f"Response: {response.status_code} - {response.text}")
#     sleep(1)  # Sleep to avoid overwhelming the server


if __name__ == "__main__":
    res = send_msg(json_msg)
    print(f"Response: {res.status_code} - {res.text}")
