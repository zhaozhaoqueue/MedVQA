import json

with open("out.json", "r") as f:
	out = [json.loads(line) for line in f][0]

# print("total sentences: ", len(out))
print("total tokens in one sentence: ", len(out["features"]))
for a in out["features"]:
	print(a["token"])
	# print(len(a["layers"]))
	for l in a["layers"]:
		print(len(l["values"]))


