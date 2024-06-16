# Copyright © 2024 Carson. All rights reserved.

import json

x =  '{ "name":"Alexander", "age":30, "city":"New York"}'

y = json.loads(x)

print(y["age"])
