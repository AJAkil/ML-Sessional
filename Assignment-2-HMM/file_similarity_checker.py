a = input()
b = input()

f1 = open(a, 'r')
f2 = open(b, 'r')

print(f1.readlines()[:15])

f1_content = [0 if data == "\"El Nino\"" else 1 for data in f1.readlines()]
f2_content = [0 if data == "\"El Nino\"" else 1 for data in f2.readlines()]
print(f1_content[:15])
print(f2_content[:15])

print(f1_content == f2_content)

counter = 0
for d1, d2 in zip(f1_content, f2_content):
    if d1 == d2:
        counter += 1

print(
    f'The files match {(counter/len(f1_content)) * 100} % and match count = {counter}'
)