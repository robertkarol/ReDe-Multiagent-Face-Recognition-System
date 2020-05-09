import random
import string


def get_random_alphanumeric_string(string_length):
    charset = string.ascii_letters + string.digits
    return ''.join((random.choice(charset) for _ in range(string_length)))
