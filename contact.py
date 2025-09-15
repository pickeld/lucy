from utiles.globals import send_request
from utiles.logger import logger
from config import config

class Contact:
    def __init__(self, payload):
        self._from = payload.get("from")
        self.participant = payload.get("participant")
        data = self.get_contact()
        logger.debug(f"Contact data retrieved: {data}")

        self.isMyContact = data.get("isMyContact", False)
        if self.isMyContact:
            self.name = data.get("name", "")
        else:
            self.name = data.get("pushname", "")

        self.number = data.get("number", "")
        self.isBusiness = data.get("isBusiness", False)
        self.is_group = data.get("isGroup", False)
        self.isUser = data.get("isUser", False)

        self.isMe = data.get("isMe", False)
        self.isBlocked = data.get("isBlocked", False)

    def get_contact(self):
        endpoint = f"/api/contacts"
        contact = self.participant if self.participant and self.participant != "out@c.us" else self._from
        params = {"contactId": contact, "session": config.waha_session_name}
        response: Dict = send_request(method="GET", endpoint=endpoint, params=params)
        return response

    def __str__(self):
        return self.__dict__.__str__()
    


if __name__ == "__main__":
    sample_payload = {'_from': '972547755011@c.us', 'participant': 'out@c.us', 'isMyContact': True, 'name': 'Me', 'number': '972547755011', 'isBusiness': False, 'is_group': False, 'isUser': True, 'isMe': True, 'isBlocked': False}
    contact = Contact(sample_payload)
    print(contact)