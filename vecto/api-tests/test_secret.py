
# Fetch Vecto config from environment
import os
secret_debug = os.environ['secret_debug']
secret_int = os.environ['secret_int']

secret_int = int(secret_int)

print(type(secret_debug))
print(type(secret_int))

print(secret_debug)
print(secret_int)

secret_debug_2 = "_" + secret_debug + "_"
secret_int_2 = "_" + str(secret_int) + "_"

print(secret_debug_2)
print(secret_int_2)

secret_int_3 =secret_int + 1
print(secret_int_3)
